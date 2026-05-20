# -*- coding: utf-8 -*-
"""训练 two-stage learned evaluator 的离线脚本。"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

import src.utils.config as runtime_config
from src.algorithms.DQN.core import DQNTwoStageLearnedEvaluator, TwoStageGraphRanker


class _TwoStageGraphDatasetBase(Dataset):
    """共享图缓存，避免同一候选反复构图。"""

    def __init__(self, graph_builder, graph_cache=None):
        self.graph_builder = graph_builder
        self.graph_cache = graph_cache if graph_cache is not None else {}

    @staticmethod
    def _record_key(record):
        return (str(record["group_id"]), int(record["proposal_idx"]))

    def _get_graph(self, record):
        cache_key = self._record_key(record)
        cached = self.graph_cache.get(cache_key)
        if cached is not None:
            return cached
        graph_data = self.graph_builder.build_data_from_record(record)
        self.graph_cache[cache_key] = graph_data
        return graph_data


class TwoStageRecordDataset(_TwoStageGraphDatasetBase):
    """stage1 主目标：判断候选是否值得进入 stage2。"""

    def __init__(self, record_samples, graph_builder, graph_cache=None):
        super().__init__(graph_builder, graph_cache=graph_cache)
        self.record_samples = list(record_samples)

    def __len__(self):
        return len(self.record_samples)

    def __getitem__(self, index):
        record, label, sample_weight = self.record_samples[index]
        return self._get_graph(record), float(label), float(sample_weight)


class TwoStagePairwiseDataset(_TwoStageGraphDatasetBase):
    """辅目标：只学习靠近 survivor 边界的关键排序。"""

    def __init__(self, pair_records, graph_builder, graph_cache=None):
        super().__init__(graph_builder, graph_cache=graph_cache)
        self.pair_records = list(pair_records)

    def __len__(self):
        return len(self.pair_records)

    def __getitem__(self, index):
        better_record, worse_record, pair_weight = self.pair_records[index]
        return (
            self._get_graph(better_record),
            self._get_graph(worse_record),
            float(pair_weight),
        )


def _collate_record_batch(batch):
    graph_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    weights = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return Batch.from_data_list(graph_list), labels, weights


def _collate_pairwise_batch(batch):
    better_graphs = [item[0] for item in batch]
    worse_graphs = [item[1] for item in batch]
    weights = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return Batch.from_data_list(better_graphs), Batch.from_data_list(worse_graphs), weights


def _load_grouped_records(dataset_path, prefer_full_labels=True):
    groups = defaultdict(list)
    with Path(dataset_path).open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "true_rank_index" not in record:
                continue
            groups[str(record["group_id"])].append(record)

    if not groups:
        return {}

    if prefer_full_labels:
        has_full_group = any(
            any(str(item.get("label_scope", "")).strip().lower() == "full" for item in records)
            for records in groups.values()
        )
        if has_full_group:
            filtered_groups = {}
            for group_id, records in groups.items():
                full_records = [
                    item
                    for item in records
                    if str(item.get("label_scope", "")).strip().lower() == "full"
                ]
                filtered_groups[group_id] = full_records if full_records else records
            groups = filtered_groups

    for group_id, records in groups.items():
        records.sort(key=lambda item: int(item["true_rank_index"]))
        groups[group_id] = records
    return groups


def _split_group_ids(group_ids, seed, val_ratio):
    rng = random.Random(int(seed))
    group_ids = list(group_ids)
    rng.shuffle(group_ids)
    if len(group_ids) <= 2 or val_ratio <= 0.0:
        return group_ids, []
    val_count = max(1, int(round(len(group_ids) * float(val_ratio))))
    if val_count >= len(group_ids):
        val_count = max(1, len(group_ids) - 1)
    return group_ids[val_count:], group_ids[:val_count]


def _infer_group_eval_budget(records):
    selected_count = sum(bool(item.get("selected_survivor", False)) for item in records)
    if selected_count > 0:
        return int(selected_count)
    return int(max(1, min(6, len(records))))


def _resolve_group_cutoffs(records, survivor_margin, boundary_band):
    eval_budget = _infer_group_eval_budget(records)
    survivor_margin = int(max(0, survivor_margin))
    boundary_band = int(max(1, boundary_band))
    positive_cutoff = min(len(records), int(eval_budget + survivor_margin))
    if positive_cutoff >= len(records):
        positive_cutoff = max(1, len(records) - 1)
    boundary_end = min(len(records), int(positive_cutoff + boundary_band))
    return int(eval_budget), int(positive_cutoff), int(boundary_end)


def _build_survivor_record_samples(
    grouped_records,
    group_ids,
    survivor_margin,
    boundary_band,
    core_positive_weight,
    far_negative_weight,
):
    record_samples = []
    for group_id in group_ids:
        records = list(grouped_records.get(group_id, []))
        if len(records) < 2:
            continue
        eval_budget, positive_cutoff, boundary_end = _resolve_group_cutoffs(
            records,
            survivor_margin,
            boundary_band,
        )
        for rank_index, record in enumerate(records):
            label = 1.0 if rank_index < positive_cutoff else 0.0
            if rank_index < eval_budget:
                sample_weight = float(core_positive_weight)
            elif rank_index < boundary_end:
                sample_weight = 1.0
            else:
                sample_weight = float(far_negative_weight)
            record_samples.append((record, label, sample_weight))
    return record_samples


def _build_boundary_pairwise_samples(
    grouped_records,
    group_ids,
    max_pairs_per_group,
    survivor_margin,
    boundary_band,
):
    pair_records = []
    for group_id in group_ids:
        records = list(grouped_records.get(group_id, []))
        if len(records) < 2:
            continue
        eval_budget, positive_cutoff, boundary_end = _resolve_group_cutoffs(
            records,
            survivor_margin,
            boundary_band,
        )
        core_positive = records[:eval_budget]
        all_positive = records[:positive_cutoff]
        margin_positive = records[eval_budget:positive_cutoff]
        boundary_negative = records[positive_cutoff:boundary_end]
        far_negative = records[boundary_end:]
        seen_pairs = set()
        group_pairs = []

        # 关键边界：所有候选 survivor 与边界外候选直接对比。
        for better_record in all_positive:
            better_rank = int(better_record["true_rank_index"])
            pair_weight = 1.5 if better_rank < eval_budget else 1.0
            for worse_record in boundary_negative:
                pair_key = (
                    int(better_record["proposal_idx"]),
                    int(worse_record["proposal_idx"]),
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                group_pairs.append((better_record, worse_record, float(pair_weight)))

        # top survivor 内部仍保留少量排序信号，避免全部坍缩成“都该留下”。
        top_window = core_positive[: min(3, len(core_positive))]
        positive_window = all_positive[: min(len(all_positive), positive_cutoff)]
        for better_index, better_record in enumerate(top_window):
            for worse_record in positive_window[better_index + 1:]:
                pair_key = (
                    int(better_record["proposal_idx"]),
                    int(worse_record["proposal_idx"]),
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                group_pairs.append((better_record, worse_record, 0.75))

        # 给最优候选补一小部分远负样本，保留粗粒度分离能力。
        if records and far_negative:
            best_record = records[0]
            for worse_record in far_negative[:2]:
                pair_key = (
                    int(best_record["proposal_idx"]),
                    int(worse_record["proposal_idx"]),
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                group_pairs.append((best_record, worse_record, 0.35))

        # 如果 margin 正样本存在，再补少量 core-vs-margin 的局部排序。
        for better_record in core_positive[:2]:
            for worse_record in margin_positive[:2]:
                better_rank = int(better_record["true_rank_index"])
                worse_rank = int(worse_record["true_rank_index"])
                if better_rank >= worse_rank:
                    continue
                pair_key = (
                    int(better_record["proposal_idx"]),
                    int(worse_record["proposal_idx"]),
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                group_pairs.append((better_record, worse_record, 0.60))

        if max_pairs_per_group > 0:
            group_pairs = group_pairs[: int(max_pairs_per_group)]
        pair_records.extend(group_pairs)
    return pair_records


def _evaluate_record_classification(model, dataloader, device):
    model.eval()
    total_count = 0
    total_correct = 0
    total_loss = 0.0
    positive_total = 0
    positive_hit = 0
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for graph_batch, labels, weights in dataloader:
            graph_batch = graph_batch.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            logits = model(graph_batch)
            loss_vec = loss_fn(logits, labels)
            loss = (loss_vec * weights).sum() / torch.clamp(weights.sum(), min=1e-8)
            total_loss += float(loss.item()) * int(labels.numel())
            predictions = (logits > 0.0).float()
            total_correct += int((predictions == labels).sum().item())
            total_count += int(labels.numel())
            positive_mask = labels > 0.5
            positive_total += int(positive_mask.sum().item())
            if positive_mask.any():
                positive_hit += int((predictions[positive_mask] > 0.5).sum().item())
    if total_count <= 0:
        return 0.0, 0.0, 0.0
    positive_recall = (positive_hit / float(positive_total)) if positive_total > 0 else 0.0
    return total_loss / float(total_count), total_correct / float(total_count), positive_recall


def _evaluate_pairwise_accuracy(model, dataloader, device):
    model.eval()
    total_pairs = 0
    total_correct = 0
    total_loss = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for better_batch, worse_batch, pair_weights in dataloader:
            better_batch = better_batch.to(device)
            worse_batch = worse_batch.to(device)
            pair_weights = pair_weights.to(device)
            better_scores = model(better_batch)
            worse_scores = model(worse_batch)
            diff = better_scores - worse_scores
            labels = torch.ones_like(diff, device=device)
            loss_vec = loss_fn(diff, labels)
            loss = (loss_vec * pair_weights).sum() / torch.clamp(pair_weights.sum(), min=1e-8)
            total_loss += float(loss.item()) * int(diff.numel())
            total_correct += int((diff > 0).sum().item())
            total_pairs += int(diff.numel())
    if total_pairs <= 0:
        return 0.0, 0.0
    return total_loss / float(total_pairs), total_correct / float(total_pairs)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_graph_score_lookup(model, graph_builder, grouped_records, group_ids, device):
    model.eval()
    score_lookup = {}
    with torch.no_grad():
        for group_id in group_ids:
            for record in grouped_records.get(group_id, []):
                graph_data = graph_builder.build_data_from_record(record)
                if graph_data is None:
                    continue
                graph_data = graph_data.to(device)
                score = float(model(graph_data).detach().cpu().item())
                score_lookup[(str(group_id), int(record["proposal_idx"]))] = score
    return score_lookup


def _evaluate_group_ranking_metrics(grouped_records, group_ids, score_lookup, topk_values=(1, 3, 5)):
    total_groups = 0
    top1_hits = 0
    proxy_top1_hits = 0
    learned_topk_hits = {int(k): 0 for k in topk_values}
    proxy_topk_hits = {int(k): 0 for k in topk_values}
    pairwise_total = 0
    pairwise_correct = 0
    proxy_pairwise_correct = 0

    for group_id in group_ids:
        records = list(grouped_records.get(group_id, []))
        if len(records) < 2:
            continue
        total_groups += 1
        true_best_key = min(records, key=lambda item: int(item["true_rank_index"]))
        true_best_idx = int(true_best_key["proposal_idx"])

        learned_ranked = sorted(
            records,
            key=lambda item: score_lookup.get((str(group_id), int(item["proposal_idx"])), float("-inf")),
            reverse=True,
        )
        proxy_ranked = sorted(
            records,
            key=lambda item: _safe_float(item.get("proxy_score", 0.0)),
            reverse=True,
        )

        if learned_ranked and int(learned_ranked[0]["proposal_idx"]) == true_best_idx:
            top1_hits += 1
        if proxy_ranked and int(proxy_ranked[0]["proposal_idx"]) == true_best_idx:
            proxy_top1_hits += 1

        for topk in topk_values:
            learned_top = {int(item["proposal_idx"]) for item in learned_ranked[: int(topk)]}
            proxy_top = {int(item["proposal_idx"]) for item in proxy_ranked[: int(topk)]}
            if true_best_idx in learned_top:
                learned_topk_hits[int(topk)] += 1
            if true_best_idx in proxy_top:
                proxy_topk_hits[int(topk)] += 1

        for better_index in range(len(records)):
            for worse_index in range(better_index + 1, len(records)):
                better = records[better_index]
                worse = records[worse_index]
                better_key = (str(group_id), int(better["proposal_idx"]))
                worse_key = (str(group_id), int(worse["proposal_idx"]))
                better_score = score_lookup.get(better_key)
                worse_score = score_lookup.get(worse_key)
                if better_score is not None and worse_score is not None:
                    pairwise_total += 1
                    if float(better_score) > float(worse_score):
                        pairwise_correct += 1
                if _safe_float(better.get("proxy_score", 0.0)) > _safe_float(worse.get("proxy_score", 0.0)):
                    proxy_pairwise_correct += 1

    metrics = {
        "groups": int(total_groups),
        "learned_top1_hit_rate": (top1_hits / float(total_groups)) if total_groups > 0 else 0.0,
        "proxy_top1_hit_rate": (proxy_top1_hits / float(total_groups)) if total_groups > 0 else 0.0,
        "learned_pairwise_accuracy": (pairwise_correct / float(pairwise_total)) if pairwise_total > 0 else 0.0,
        "proxy_pairwise_accuracy": (
            proxy_pairwise_correct / float(pairwise_total)
            if pairwise_total > 0 else 0.0
        ),
    }
    for topk in topk_values:
        metrics[f"learned_top{int(topk)}_recall"] = (
            learned_topk_hits[int(topk)] / float(total_groups)
            if total_groups > 0 else 0.0
        )
        metrics[f"proxy_top{int(topk)}_recall"] = (
            proxy_topk_hits[int(topk)] / float(total_groups)
            if total_groups > 0 else 0.0
        )
    return metrics


def parse_args():
    defaults = getattr(runtime_config, "ELP_RUNTIME_DEFAULTS", {})
    parser = argparse.ArgumentParser(description="训练 two-stage learned evaluator。")
    parser.add_argument(
        "--dataset-path",
        default=str(defaults.get("two_stage_learned_evaluator_dataset_path", "")),
        help="two-stage 数据集 JSONL 路径。",
    )
    parser.add_argument(
        "--output-path",
        default=str(defaults.get("two_stage_learned_evaluator_model_path", "")),
        help="训练完成后模型 checkpoint 保存路径。",
    )
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=32, help="pairwise 批大小。")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率。")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(defaults.get("two_stage_learned_evaluator_hidden_dim", 64)),
        help="图网络隐藏维度。",
    )
    parser.add_argument(
        "--message-steps",
        type=int,
        default=int(defaults.get("two_stage_learned_evaluator_message_steps", 2)),
        help="图消息传递层数。",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=float(defaults.get("two_stage_learned_evaluator_dropout", 0.05)),
        help="头部 dropout。",
    )
    parser.add_argument("--seed", type=int, default=20260420, help="随机种子。")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集 group 占比。")
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=24,
        help="每个 proposal group 最多保留多少个关键边界 pair。",
    )
    parser.add_argument(
        "--survivor-margin",
        type=int,
        default=2,
        help="在原 eval budget 之外，额外保留多少个候选作为 stage1 正样本。",
    )
    parser.add_argument(
        "--boundary-band",
        type=int,
        default=3,
        help="在 survivor 边界外，额外关注多少个紧邻负样本。",
    )
    parser.add_argument(
        "--core-positive-weight",
        type=float,
        default=1.8,
        help="核心 survivor 正样本的分类权重。",
    )
    parser.add_argument(
        "--far-negative-weight",
        type=float,
        default=0.20,
        help="远离 survivor 边界的负样本分类权重。",
    )
    parser.add_argument(
        "--classification-weight",
        type=float,
        default=1.0,
        help="survivor 二分类主损失权重。",
    )
    parser.add_argument(
        "--ranking-weight",
        type=float,
        default=0.35,
        help="关键边界 pairwise 辅损失权重。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="训练设备。",
    )
    parser.add_argument(
        "--allow-survivor-only",
        action="store_true",
        help="当 full label 不足时允许使用 survivor label 数据。",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="训练中间 checkpoint 保存路径；留空时按 output-path 自动推导。",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="恢复训练时使用的 checkpoint 路径；留空时若默认 checkpoint 存在则自动恢复。",
    )
    parser.add_argument("--save-every", type=int, default=1, help="每隔多少个 epoch 保存一次 checkpoint。")
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="忽略已有 checkpoint，从头开始训练。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    grouped_records = _load_grouped_records(
        dataset_path,
        prefer_full_labels=not bool(args.allow_survivor_only),
    )
    if not grouped_records:
        raise RuntimeError("数据集为空，无法训练 two-stage learned evaluator。")

    train_group_ids, val_group_ids = _split_group_ids(
        list(grouped_records.keys()),
        args.seed,
        args.val_ratio,
    )
    if not train_group_ids:
        raise RuntimeError("训练集为空，请增加数据后再训练。")

    train_record_samples = _build_survivor_record_samples(
        grouped_records,
        train_group_ids,
        args.survivor_margin,
        args.boundary_band,
        args.core_positive_weight,
        args.far_negative_weight,
    )
    val_record_samples = _build_survivor_record_samples(
        grouped_records,
        val_group_ids,
        args.survivor_margin,
        args.boundary_band,
        args.core_positive_weight,
        args.far_negative_weight,
    )
    train_pairs = _build_boundary_pairwise_samples(
        grouped_records,
        train_group_ids,
        args.max_pairs_per_group,
        args.survivor_margin,
        args.boundary_band,
    )
    val_pairs = _build_boundary_pairwise_samples(
        grouped_records,
        val_group_ids,
        args.max_pairs_per_group,
        args.survivor_margin,
        args.boundary_band,
    )
    if not train_record_samples:
        raise RuntimeError("没有生成任何 stage1 survivor 训练样本。")
    if not train_pairs:
        raise RuntimeError("没有生成任何关键边界 pairwise 训练样本。")

    graph_builder = DQNTwoStageLearnedEvaluator()
    shared_graph_cache = {}
    train_record_dataset = TwoStageRecordDataset(
        train_record_samples,
        graph_builder,
        graph_cache=shared_graph_cache,
    )
    val_record_dataset = (
        TwoStageRecordDataset(
            val_record_samples,
            graph_builder,
            graph_cache=shared_graph_cache,
        )
        if val_record_samples else None
    )
    train_pair_dataset = TwoStagePairwiseDataset(
        train_pairs,
        graph_builder,
        graph_cache=shared_graph_cache,
    )
    val_pair_dataset = (
        TwoStagePairwiseDataset(
            val_pairs,
            graph_builder,
            graph_cache=shared_graph_cache,
        )
        if val_pairs else None
    )

    train_record_loader = DataLoader(
        train_record_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        collate_fn=_collate_record_batch,
    )
    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        collate_fn=_collate_pairwise_batch,
    )
    val_record_loader = None
    if val_record_dataset is not None and len(val_record_dataset) > 0:
        val_record_loader = DataLoader(
            val_record_dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            collate_fn=_collate_record_batch,
        )
    val_pair_loader = None
    if val_pair_dataset is not None and len(val_pair_dataset) > 0:
        val_pair_loader = DataLoader(
            val_pair_dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            collate_fn=_collate_pairwise_batch,
        )

    model = TwoStageGraphRanker(
        node_dim=graph_builder.node_dim,
        global_dim=graph_builder.global_dim,
        hidden_dim=args.hidden_dim,
        message_steps=args.message_steps,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    record_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    pair_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if str(args.checkpoint_path).strip()
        else output_path.with_name(f"{output_path.stem}_checkpoint.pt")
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = None
    if not bool(args.fresh_start):
        if str(args.resume_from).strip():
            resume_path = Path(args.resume_from)
        elif checkpoint_path.exists():
            resume_path = checkpoint_path

    start_epoch = 1
    best_metric = None
    best_state_dict = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"恢复 checkpoint 不存在: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        best_metric = resume_checkpoint.get("best_metric")
        best_state_dict = resume_checkpoint.get("best_state_dict")
        if best_state_dict is None:
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        print(f"resumed_from={resume_path}")
        print(f"resume_epoch={start_epoch}")

    for epoch_idx in range(int(start_epoch), int(args.epochs) + 1):
        model.train()
        train_record_loss_sum = 0.0
        train_record_count = 0
        train_record_correct = 0
        train_record_positive_total = 0
        train_record_positive_hit = 0
        for graph_batch, labels, sample_weights in train_record_loader:
            graph_batch = graph_batch.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)
            logits = model(graph_batch)
            loss_vec = record_loss_fn(logits, labels)
            loss = (loss_vec * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1e-8)
            loss = float(args.classification_weight) * loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            predictions = (logits > 0.0).float()
            positive_mask = labels > 0.5

            train_record_loss_sum += float(loss.item()) * int(labels.numel())
            train_record_count += int(labels.numel())
            train_record_correct += int((predictions == labels).sum().item())
            train_record_positive_total += int(positive_mask.sum().item())
            if positive_mask.any():
                train_record_positive_hit += int((predictions[positive_mask] > 0.5).sum().item())

        train_pair_loss_sum = 0.0
        train_pair_count = 0
        train_pair_correct = 0
        for better_batch, worse_batch, pair_weights in train_pair_loader:
            better_batch = better_batch.to(device)
            worse_batch = worse_batch.to(device)
            pair_weights = pair_weights.to(device)
            better_scores = model(better_batch)
            worse_scores = model(worse_batch)
            diff = better_scores - worse_scores
            labels = torch.ones_like(diff, device=device)
            loss_vec = pair_loss_fn(diff, labels)
            loss = (loss_vec * pair_weights).sum() / torch.clamp(pair_weights.sum(), min=1e-8)
            loss = float(args.ranking_weight) * loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_pair_loss_sum += float(loss.item()) * int(diff.numel())
            train_pair_count += int(diff.numel())
            train_pair_correct += int((diff > 0).sum().item())

        train_record_loss = train_record_loss_sum / float(max(1, train_record_count))
        train_record_acc = train_record_correct / float(max(1, train_record_count))
        train_positive_recall = (
            train_record_positive_hit / float(train_record_positive_total)
            if train_record_positive_total > 0 else 0.0
        )
        train_pair_loss = train_pair_loss_sum / float(max(1, train_pair_count))
        train_pair_acc = train_pair_correct / float(max(1, train_pair_count))

        if val_record_loader is not None and val_pair_loader is not None:
            val_record_loss, val_record_acc, val_positive_recall = _evaluate_record_classification(
                model,
                val_record_loader,
                device,
            )
            val_pair_loss, val_pair_acc = _evaluate_pairwise_accuracy(model, val_pair_loader, device)
            metric = 0.70 * float(val_positive_recall) + 0.30 * float(val_pair_acc)
            print(
                f"epoch={epoch_idx:03d} "
                f"train_cls_loss={train_record_loss:.4f} train_cls_acc={train_record_acc:.4f} "
                f"train_pos_recall={train_positive_recall:.4f} "
                f"train_rank_loss={train_pair_loss:.4f} train_rank_acc={train_pair_acc:.4f} "
                f"val_cls_loss={val_record_loss:.4f} val_cls_acc={val_record_acc:.4f} "
                f"val_pos_recall={val_positive_recall:.4f} "
                f"val_rank_loss={val_pair_loss:.4f} val_rank_acc={val_pair_acc:.4f}"
            )
        else:
            val_record_loss, val_record_acc, val_positive_recall = 0.0, 0.0, 0.0
            val_pair_loss, val_pair_acc = 0.0, 0.0
            metric = 0.70 * float(train_positive_recall) + 0.30 * float(train_pair_acc)
            print(
                f"epoch={epoch_idx:03d} "
                f"train_cls_loss={train_record_loss:.4f} train_cls_acc={train_record_acc:.4f} "
                f"train_pos_recall={train_positive_recall:.4f} "
                f"train_rank_loss={train_pair_loss:.4f} train_rank_acc={train_pair_acc:.4f}"
            )

        if best_metric is None or metric >= best_metric:
            best_metric = metric
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if int(args.save_every) > 0 and epoch_idx % int(args.save_every) == 0:
            checkpoint = {
                "epoch": int(epoch_idx),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": (float(best_metric) if best_metric is not None else None),
                "best_state_dict": best_state_dict,
                "node_dim": int(graph_builder.node_dim),
                "global_dim": int(graph_builder.global_dim),
                "hidden_dim": int(args.hidden_dim),
                "message_steps": int(args.message_steps),
                "dropout": float(args.dropout),
                "train_groups": int(len(train_group_ids)),
                "val_groups": int(len(val_group_ids)),
                "train_records": int(len(train_record_samples)),
                "val_records": int(len(val_record_samples)),
                "train_pairs": int(len(train_pairs)),
                "val_pairs": int(len(val_pairs)),
                "dataset_path": str(dataset_path),
                "output_path": str(output_path),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"saved_checkpoint={checkpoint_path}")
            print(f"checkpoint_epoch={epoch_idx}")

    if best_state_dict is None:
        raise RuntimeError("训练失败，未生成有效模型参数。")

    checkpoint = {
        "model_state_dict": best_state_dict,
        "node_dim": int(graph_builder.node_dim),
        "global_dim": int(graph_builder.global_dim),
        "hidden_dim": int(args.hidden_dim),
        "message_steps": int(args.message_steps),
        "dropout": float(args.dropout),
        "train_groups": int(len(train_group_ids)),
        "val_groups": int(len(val_group_ids)),
        "train_records": int(len(train_record_samples)),
        "val_records": int(len(val_record_samples)),
        "train_pairs": int(len(train_pairs)),
        "val_pairs": int(len(val_pairs)),
        "best_metric": float(best_metric),
    }
    torch.save(checkpoint, output_path)
    print(f"saved_model={output_path}")

    evaluation_model = TwoStageGraphRanker(
        node_dim=graph_builder.node_dim,
        global_dim=graph_builder.global_dim,
        hidden_dim=args.hidden_dim,
        message_steps=args.message_steps,
        dropout=args.dropout,
    ).to(device)
    evaluation_model.load_state_dict(best_state_dict)
    learned_train_lookup = _build_graph_score_lookup(
        evaluation_model,
        graph_builder,
        grouped_records,
        train_group_ids,
        device,
    )
    train_metrics = _evaluate_group_ranking_metrics(
        grouped_records,
        train_group_ids,
        learned_train_lookup,
    )
    print(
        "train_eval "
        f"groups={train_metrics['groups']} "
        f"learned_top1={train_metrics['learned_top1_hit_rate']:.4f} "
        f"proxy_top1={train_metrics['proxy_top1_hit_rate']:.4f} "
        f"learned_top3={train_metrics['learned_top3_recall']:.4f} "
        f"proxy_top3={train_metrics['proxy_top3_recall']:.4f} "
        f"learned_top5={train_metrics['learned_top5_recall']:.4f} "
        f"proxy_top5={train_metrics['proxy_top5_recall']:.4f} "
        f"learned_pairwise={train_metrics['learned_pairwise_accuracy']:.4f} "
        f"proxy_pairwise={train_metrics['proxy_pairwise_accuracy']:.4f}"
    )
    if val_group_ids:
        learned_val_lookup = _build_graph_score_lookup(
            evaluation_model,
            graph_builder,
            grouped_records,
            val_group_ids,
            device,
        )
        val_metrics = _evaluate_group_ranking_metrics(
            grouped_records,
            val_group_ids,
            learned_val_lookup,
        )
        print(
            "val_eval "
            f"groups={val_metrics['groups']} "
            f"learned_top1={val_metrics['learned_top1_hit_rate']:.4f} "
            f"proxy_top1={val_metrics['proxy_top1_hit_rate']:.4f} "
            f"learned_top3={val_metrics['learned_top3_recall']:.4f} "
            f"proxy_top3={val_metrics['proxy_top3_recall']:.4f} "
            f"learned_top5={val_metrics['learned_top5_recall']:.4f} "
            f"proxy_top5={val_metrics['proxy_top5_recall']:.4f} "
            f"learned_pairwise={val_metrics['learned_pairwise_accuracy']:.4f} "
            f"proxy_pairwise={val_metrics['proxy_pairwise_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
