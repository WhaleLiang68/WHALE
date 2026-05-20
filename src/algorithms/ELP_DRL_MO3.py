import copy
import csv
import json
import math
import os
from pathlib import Path

import gym
import numpy as np
from loguru import logger

import src
from src.algorithms.ELP_DRL_MO import ELP as BaseMOELP
from src.algorithms.ELP_DRL_MO import _save_experiment_row
from src.algorithms.ELP_DRL_Standard import _get_initial_solution_energy, _set_global_seed
from src.utils.MO_FBSUtil import MO_FBSUtil


class ELP(BaseMOELP):
    """MO3: ?? MO ?????????????????????"""

    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0, archive_limit=64, objective_weights=None):
        super().__init__(env, gbest, T, G=G, t_max=t_max, k=k, archive_limit=archive_limit, objective_weights=objective_weights)

        # ?????????????????????????
        self.archive_require_candidate_retained = True

        # ???? spacing ???HV ????????????????????
        self.archive_spacing_guard_when_full = self._parse_env_flag("ELP_MO3_ARCHIVE_SPACING_GUARD", True)
        self.archive_spacing_guard_rel_tol = max(
            0.0,
            float(os.getenv("ELP_MO3_ARCHIVE_SPACING_GUARD_REL_TOL", "0.12")),
        )
        self.archive_spacing_guard_hv_gain_rel = max(
            0.0,
            float(os.getenv("ELP_MO3_ARCHIVE_SPACING_GUARD_HV_GAIN_REL", "0.03")),
        )

        # ????????????? HV ????????????????????????
        self.mo3_boundary_hv_gain_rel = max(
            0.0,
            float(os.getenv("ELP_MO3_BOUNDARY_HV_GAIN_REL", "0.025")),
        )
        self.mo3_boundary_rep_slack = max(
            0.0,
            float(os.getenv("ELP_MO3_BOUNDARY_REP_SLACK", "0.02")),
        )
        self.mo3_boundary_bonus = max(
            0.0,
            float(os.getenv("ELP_MO3_BOUNDARY_BONUS", "0.18")),
        )
        self._last_mo3_archive_feedback = {}
        self._reset_mo3_telemetry()

    def _run_impl(self):
        self._reset_mo3_telemetry()
        os.environ.setdefault("ELP_EXP_ALGORITHM", "ELP_DRL_MO3")
        os.environ.setdefault(
            "ELP_EXP_REMARK",
            "MO strong skeleton with retained-candidate semantics and light boundary-aware archive shaping",
        )
        result = super()._run_impl()
        self._finalize_mo3_telemetry()
        return result

    def _reset_mo3_telemetry(self):
        # ?? MO3 ????????????????????????
        self._mo3_telemetry = {
            "feasibleObserveCalls": 0,
            "archivePreviewInsertions": 0,
            "archiveChanged": 0,
            "candidateRetainedRejects": 0,
            "candidateRetainedAnomalies": 0,
            "spacingGuardChecks": 0,
            "spacingGuardRejects": 0,
            "spacingGuardHvRelGainSum": 0.0,
            "spacingGuardSpacingRelChangeSum": 0.0,
            "spacingGuardRepScoreDeltaSum": 0.0,
            "boundaryBonusTriggers": 0,
            "boundaryBonusRewardTotal": 0.0,
            "boundaryHvRelGainSum": 0.0,
            "boundaryRepScoreDeltaSum": 0.0,
        }

    def _record_mo3_counter(self, key, amount=1.0):
        self._mo3_telemetry[key] = float(self._mo3_telemetry.get(key, 0.0) or 0.0) + float(amount)

    def _compute_archive_rep_score(self, archive, ideal, nadir):
        if not archive:
            return math.inf
        _rep_solution, rep_score, _rep_index = MO_FBSUtil.select_representative_solution(
            archive,
            ideal=ideal,
            nadir=nadir,
            weights=self.mo_weights,
        )
        return float(rep_score) if np.isfinite(rep_score) else math.inf

    def _preview_archive_without_mo3_guards(self, before_archive, solution, ideal, nadir):
        return MO_FBSUtil.update_pareto_archive(
            before_archive,
            solution,
            max_size=self.archive_limit,
            clone_fn=copy.deepcopy,
            quality_gate_when_full=self.archive_quality_gate_when_full,
            quality_hv_tol=self.archive_quality_hv_tol,
            quality_spacing_tol=self.archive_quality_spacing_tol,
            spacing_guard_when_full=False,
            spacing_guard_rel_tol=0.0,
            spacing_guard_hv_gain_rel=0.0,
            require_candidate_retained=False,
            ideal=ideal,
            nadir=nadir,
        )

    def _build_mo3_summary_fields(self):
        telemetry = dict(getattr(self, "_mo3_telemetry", {}) or {})
        spacing_guard_rejects = int(telemetry.get("spacingGuardRejects", 0) or 0)
        boundary_triggers = int(telemetry.get("boundaryBonusTriggers", 0) or 0)
        summary_fields = {
            "mo3FeasibleObserveCalls": int(telemetry.get("feasibleObserveCalls", 0) or 0),
            "mo3ArchivePreviewInsertions": int(telemetry.get("archivePreviewInsertions", 0) or 0),
            "mo3ArchiveChanged": int(telemetry.get("archiveChanged", 0) or 0),
            "mo3CandidateRetainedRejects": int(telemetry.get("candidateRetainedRejects", 0) or 0),
            "mo3CandidateRetainedAnomalies": int(telemetry.get("candidateRetainedAnomalies", 0) or 0),
            "mo3SpacingGuardChecks": int(telemetry.get("spacingGuardChecks", 0) or 0),
            "mo3SpacingGuardRejects": spacing_guard_rejects,
            "mo3BoundaryBonusTriggers": boundary_triggers,
            "mo3BoundaryBonusRewardTotal": float(telemetry.get("boundaryBonusRewardTotal", 0.0) or 0.0),
            "mo3SpacingGuardHvRelGainAvg": None,
            "mo3SpacingGuardSpacingRelChangeAvg": None,
            "mo3SpacingGuardRepScoreDeltaAvg": None,
            "mo3BoundaryHvRelGainAvg": None,
            "mo3BoundaryRepScoreDeltaAvg": None,
        }
        if spacing_guard_rejects > 0:
            summary_fields["mo3SpacingGuardHvRelGainAvg"] = float(
                telemetry.get("spacingGuardHvRelGainSum", 0.0) or 0.0
            ) / float(spacing_guard_rejects)
            summary_fields["mo3SpacingGuardSpacingRelChangeAvg"] = float(
                telemetry.get("spacingGuardSpacingRelChangeSum", 0.0) or 0.0
            ) / float(spacing_guard_rejects)
            summary_fields["mo3SpacingGuardRepScoreDeltaAvg"] = float(
                telemetry.get("spacingGuardRepScoreDeltaSum", 0.0) or 0.0
            ) / float(spacing_guard_rejects)
        if boundary_triggers > 0:
            summary_fields["mo3BoundaryHvRelGainAvg"] = float(
                telemetry.get("boundaryHvRelGainSum", 0.0) or 0.0
            ) / float(boundary_triggers)
            summary_fields["mo3BoundaryRepScoreDeltaAvg"] = float(
                telemetry.get("boundaryRepScoreDeltaSum", 0.0) or 0.0
            ) / float(boundary_triggers)
        return summary_fields

    def _rewrite_summary_csv_row(self, summary_csv_path, run_id, extra_fields):
        summary_csv_path = Path(summary_csv_path)
        if not summary_csv_path.exists():
            return
        with summary_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0].keys()) if rows else []
        if not rows:
            return
        for key in extra_fields.keys():
            if key not in fieldnames:
                fieldnames.append(key)
        updated = False
        for row in rows:
            if str(row.get("runId", "")) != str(run_id):
                continue
            for key, value in extra_fields.items():
                row[key] = value
            updated = True
            break
        if not updated:
            return
        with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _finalize_mo3_telemetry(self):
        if not getattr(self, "mo_run_summary", None):
            return
        summary_fields = self._build_mo3_summary_fields()
        self.mo_run_summary.update(summary_fields)
        if isinstance(getattr(self, "last_run_payload", None), dict):
            self.last_run_payload.update(summary_fields)

        repo_root = getattr(getattr(self, "mo_recorder", None), "repo_root", None)
        run_summary_path = self.mo_run_summary.get("runSummaryPath")
        if repo_root is not None and run_summary_path:
            run_summary_file = Path(repo_root) / str(run_summary_path)
            if run_summary_file.exists():
                payload = json.loads(run_summary_file.read_text(encoding="utf-8"))
                payload.update(summary_fields)
                run_summary_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_csv_path = getattr(getattr(self, "mo_recorder", None), "summary_csv_path", None)
        run_id = self.mo_run_summary.get("runId")
        if summary_csv_path is not None and run_id:
            self._rewrite_summary_csv_row(summary_csv_path, run_id, summary_fields)

    def _observe_feasible_state(self, solution):
        if not getattr(solution, "current_is_feasible", False):
            self._last_mo3_archive_feedback = {}
            return super()._observe_feasible_state(solution)

        self._record_mo3_counter("feasibleObserveCalls", 1)
        before_archive = list(self.pareto_archive)
        before_ideal = self.mo_ideal
        before_nadir = self.mo_nadir
        before_hv = 0.0
        before_spacing = 0.0
        rep_score_before = math.inf
        if before_archive:
            before_hv = float(
                MO_FBSUtil.archive_hypervolume(before_archive, ideal=before_ideal, nadir=before_nadir) or 0.0
            )
            before_spacing = float(
                MO_FBSUtil.archive_spacing(before_archive, ideal=before_ideal, nadir=before_nadir) or 0.0
            )
            _, rep_score_before, _ = MO_FBSUtil.select_representative_solution(
                before_archive,
                ideal=before_ideal,
                nadir=before_nadir,
                weights=self.mo_weights,
            )

        preview_archive, preview_inserted, _preview_removed = self._preview_archive_without_mo3_guards(
            before_archive,
            solution,
            before_ideal,
            before_nadir,
        )
        preview_candidate_retained = bool(
            preview_inserted
            and any(MO_FBSUtil._duplicate_objectives(solution, existing, atol=1e-9) for existing in preview_archive)
        )
        if preview_inserted:
            self._record_mo3_counter("archivePreviewInsertions", 1)

        full_before_insert = bool(self.archive_limit is not None and len(before_archive) >= int(self.archive_limit))
        preview_hv = before_hv
        preview_spacing = before_spacing
        rep_score_preview = rep_score_before
        if preview_inserted:
            preview_hv = float(
                MO_FBSUtil.archive_hypervolume(preview_archive, ideal=before_ideal, nadir=before_nadir) or 0.0
            )
            preview_spacing = float(
                MO_FBSUtil.archive_spacing(preview_archive, ideal=before_ideal, nadir=before_nadir) or 0.0
            )
            rep_score_preview = self._compute_archive_rep_score(preview_archive, before_ideal, before_nadir)

        archive_changed = bool(super()._observe_feasible_state(solution))
        if archive_changed:
            self._record_mo3_counter("archiveChanged", 1)

        candidate_retained = any(
            MO_FBSUtil._duplicate_objectives(solution, existing, atol=1e-9) for existing in self.pareto_archive
        )
        after_hv = float(
            MO_FBSUtil.archive_hypervolume(self.pareto_archive, ideal=self.mo_ideal, nadir=self.mo_nadir) or 0.0
        )
        after_spacing = float(
            MO_FBSUtil.archive_spacing(self.pareto_archive, ideal=self.mo_ideal, nadir=self.mo_nadir) or 0.0
        )
        rep_score_after = float(self.representative_decision_score)

        if before_hv <= 1e-12:
            hv_rel_gain = 1.0 if after_hv > before_hv + 1e-12 else 0.0
        else:
            hv_rel_gain = (after_hv - before_hv) / before_hv

        spacing_rel_change = 0.0
        if before_archive and before_spacing > 1e-12:
            spacing_rel_change = (after_spacing - before_spacing) / before_spacing

        if preview_inserted and not archive_changed:
            if bool(self.archive_require_candidate_retained) and not preview_candidate_retained:
                self._record_mo3_counter("candidateRetainedRejects", 1)
                self._record_mo_event(
                    "mo3_candidate_retained_reject",
                    candidateDecisionScore=self._safe_float(getattr(solution, "decision_score", None)),
                    candidateMhc=self._safe_float(getattr(solution, "MHC", None)),
                    candidateCr=self._safe_float(getattr(solution, "CR", None)),
                    candidateDr=self._safe_float(getattr(solution, "DR", None)),
                    candidateAr=self._safe_float(getattr(solution, "AR", None)),
                    previewArchiveSize=int(len(preview_archive)),
                )
            elif bool(self.archive_spacing_guard_when_full) and full_before_insert and preview_candidate_retained:
                spacing_guard_hv_rel_gain = 0.0
                if before_hv <= 1e-12:
                    spacing_guard_hv_rel_gain = 1.0 if preview_hv > before_hv + 1e-12 else 0.0
                else:
                    spacing_guard_hv_rel_gain = (preview_hv - before_hv) / before_hv
                spacing_guard_spacing_rel_change = 0.0
                if before_archive and before_spacing > 1e-12:
                    spacing_guard_spacing_rel_change = (preview_spacing - before_spacing) / before_spacing
                spacing_guard_rep_delta = 0.0
                if np.isfinite(rep_score_before) and np.isfinite(rep_score_preview):
                    spacing_guard_rep_delta = rep_score_preview - rep_score_before
                self._record_mo3_counter("spacingGuardChecks", 1)
                self._record_mo3_counter("spacingGuardRejects", 1)
                self._record_mo3_counter("spacingGuardHvRelGainSum", spacing_guard_hv_rel_gain)
                self._record_mo3_counter("spacingGuardSpacingRelChangeSum", spacing_guard_spacing_rel_change)
                self._record_mo3_counter("spacingGuardRepScoreDeltaSum", spacing_guard_rep_delta)
                self._record_mo_event(
                    "mo3_spacing_guard_reject",
                    candidateDecisionScore=self._safe_float(getattr(solution, "decision_score", None)),
                    candidateMhc=self._safe_float(getattr(solution, "MHC", None)),
                    candidateCr=self._safe_float(getattr(solution, "CR", None)),
                    candidateDr=self._safe_float(getattr(solution, "DR", None)),
                    candidateAr=self._safe_float(getattr(solution, "AR", None)),
                    hvRelGain=self._safe_float(spacing_guard_hv_rel_gain),
                    spacingRelChange=self._safe_float(spacing_guard_spacing_rel_change),
                    repScoreBefore=self._safe_float(rep_score_before),
                    repScorePreview=self._safe_float(rep_score_preview),
                    repScoreDelta=self._safe_float(spacing_guard_rep_delta),
                )

        # ????????????????????????????
        boundary_useful = bool(
            archive_changed
            and candidate_retained
            and hv_rel_gain >= float(self.mo3_boundary_hv_gain_rel)
            and (
                (not np.isfinite(rep_score_before))
                or rep_score_after <= rep_score_before + float(self.mo3_boundary_rep_slack)
            )
        )
        if archive_changed and not candidate_retained:
            self._record_mo3_counter("candidateRetainedAnomalies", 1)
            self._record_mo_event(
                "mo3_candidate_retained_anomaly",
                candidateDecisionScore=self._safe_float(getattr(solution, "decision_score", None)),
                candidateMhc=self._safe_float(getattr(solution, "MHC", None)),
                candidateCr=self._safe_float(getattr(solution, "CR", None)),
                candidateDr=self._safe_float(getattr(solution, "DR", None)),
                candidateAr=self._safe_float(getattr(solution, "AR", None)),
            )

        self._last_mo3_archive_feedback = {
            "archive_changed": bool(archive_changed),
            "candidate_retained": bool(candidate_retained),
            "boundary_useful": bool(boundary_useful),
            "hv_rel_gain": float(hv_rel_gain),
            "spacing_rel_change": float(spacing_rel_change),
            "rep_score_before": float(rep_score_before) if np.isfinite(rep_score_before) else math.inf,
            "rep_score_after": float(rep_score_after) if np.isfinite(rep_score_after) else math.inf,
        }
        return archive_changed

    def _compute_transition_reward(
        self,
        previous_cost,
        next_cost,
        previous_d_inf,
        next_d_inf,
        previous_best_feasible,
        accept,
    ):
        reward = super()._compute_transition_reward(
            previous_cost,
            next_cost,
            previous_d_inf,
            next_d_inf,
            previous_best_feasible,
            accept,
        )
        feedback = dict(getattr(self, "_last_mo3_archive_feedback", {}) or {})
        if accept and bool(feedback.get("boundary_useful", False)):
            reward += float(self.mo3_boundary_bonus)
            rep_score_before = feedback.get("rep_score_before", math.inf)
            rep_score_after = feedback.get("rep_score_after", math.inf)
            rep_score_delta = 0.0
            if np.isfinite(rep_score_before) and np.isfinite(rep_score_after):
                rep_score_delta = float(rep_score_after - rep_score_before)
            self._record_mo3_counter("boundaryBonusTriggers", 1)
            self._record_mo3_counter("boundaryBonusRewardTotal", float(self.mo3_boundary_bonus))
            self._record_mo3_counter("boundaryHvRelGainSum", float(feedback.get("hv_rel_gain", 0.0) or 0.0))
            self._record_mo3_counter("boundaryRepScoreDeltaSum", rep_score_delta)
            self._record_mo_event(
                "mo3_boundary_bonus",
                bonus=self._safe_float(self.mo3_boundary_bonus),
                hvRelGain=self._safe_float(feedback.get("hv_rel_gain", 0.0)),
                spacingRelChange=self._safe_float(feedback.get("spacing_rel_change", 0.0)),
                repScoreBefore=self._safe_float(rep_score_before),
                repScoreAfter=self._safe_float(rep_score_after),
                repScoreDelta=self._safe_float(rep_score_delta),
                candidateDecisionScore=self._safe_float(getattr(self.s, "decision_score", None)),
            )
        return float(np.clip(reward, -3.0, 3.0))


if __name__ == "__main__":
    def _parse_env_int(name, default):
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        try:
            return int(raw.strip())
        except Exception:
            return int(default)

    def _parse_env_float(name, default):
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        try:
            return float(raw.strip())
        except Exception:
            return float(default)

    def _parse_env_flag(name, default):
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on", "y"}

    exp_instance = os.getenv("ELP_EXP_INSTANCE", "Du62")
    baseline_algo = os.getenv("ELP_MO_BASELINE_ALGO", "").strip().lower()
    baseline_enabled = baseline_algo in {"nsga2", "moead", "spea2"}
    default_algorithm = f"MO_BASELINE_{baseline_algo.upper()}" if baseline_enabled else "ELP_DRL_MO3"
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", default_algorithm)
    default_remark = (
        "MO baseline on action-sequence encoding"
        if baseline_enabled
        else "MO strong skeleton with retained-candidate semantics and light boundary-aware archive shaping"
    )
    exp_remark = os.getenv("ELP_EXP_REMARK", default_remark)
    exp_number = _parse_env_int("ELP_EXP_NUMBER", 30)
    is_exp = _parse_env_flag("ELP_IS_EXP", True)

    G = _parse_env_int("ELP_G", 1000)
    t_max = _parse_env_int("ELP_T_MAX", 300)
    T_initial = _parse_env_float("ELP_T_INITIAL", 10000.0)
    k_hist = _parse_env_float("ELP_K_HIST", 10.0)
    base_seed = _parse_env_int("ELP_BASE_SEED", 20260427)

    baseline_population = _parse_env_int("ELP_MO_BASELINE_POP", 64)
    baseline_generations = _parse_env_int("ELP_MO_BASELINE_GEN", 80)
    baseline_sequence_length = _parse_env_int("ELP_MO_BASELINE_SEQ_LEN", t_max)

    def _run_once(run_index):
        run_seed = int(base_seed + run_index)
        strict_determinism = _set_global_seed(run_seed)
        logger.info(f"Experiment seed: {run_seed} | strict_determinism: {strict_determinism}")
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        try:
            env.reset(seed=run_seed)
        except TypeError:
            env.reset()
        except Exception:
            env.reset()
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        initial_gbest = copy.deepcopy(base_env)
        logger.info(f"Initial solution energy: {_get_initial_solution_energy(base_env)}")
        solver = ELP(
            env=base_env,
            gbest=initial_gbest,
            T=T_initial,
            G=G,
            t_max=t_max,
            k=k_hist,
        )
        if baseline_enabled:
            return solver, solver.run_moea_baseline(
                algorithm_name=baseline_algo,
                population_size=baseline_population,
                generations=baseline_generations,
                sequence_length=baseline_sequence_length,
                seed=run_seed,
            )
        return solver, solver.run()

    if is_exp:
        for i in range(exp_number):
            logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
            try:
                elp_solver, result_tuple = _run_once(i)
                total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
                logger.info(f"Experiment {i + 1} complete | representative decision score: {best_energy}")
                if not baseline_enabled:
                    for telemetry_line in elp_solver.format_action_telemetry():
                        logger.info(f"Telemetry | {telemetry_line}")
                _save_experiment_row(
                    exp_instance,
                    exp_algorithm,
                    exp_remark,
                    total_iter,
                    is_valid,
                    best_sol,
                    best_energy,
                    start,
                    end,
                    fast,
                    elp_solver,
                )
            except Exception as exc:
                logger.exception(f"Experiment {i + 1} failed: {exc}")
    else:
        elp_solver, result_tuple = _run_once(0)
        total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
        print(f"Single run complete | representative decision score: {best_energy}")
        if not baseline_enabled:
            for telemetry_line in elp_solver.format_action_telemetry():
                print(f"Telemetry | {telemetry_line}")
        _save_experiment_row(
            exp_instance,
            exp_algorithm,
            exp_remark,
            total_iter,
            is_valid,
            best_sol,
            best_energy,
            start,
            end,
            fast,
            elp_solver,
        )
