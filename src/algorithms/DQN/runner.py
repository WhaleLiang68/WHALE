import copy
import datetime
import os
from dataclasses import dataclass

import numpy as np

from .core import StandardDQNAgent, StandardQLearningAgent


@dataclass
class DQNTrainingStepResult:
    """单步训练 transition 的摘要。"""

    action_table_idx: int
    real_action_idx: int
    reward: float
    step_improved: bool
    archive_improved: bool
    gbest_recomputed: bool
    trigger_local_search: bool
    local_search_improved: bool


class DQNTransitionEngine:
    """将主循环中的单步 transition 逻辑从大文件中剥离。"""

    def execute_training_step(self, runtime, agent, step_idx):
        update_epsilon_schedule = getattr(agent, "update_epsilon_schedule", None)
        if callable(update_epsilon_schedule):
            update_epsilon_schedule(runtime.current_global_step)
        record_selection_action_mode = getattr(runtime, "_record_selection_action_mode", None)
        if callable(record_selection_action_mode):
            record_selection_action_mode(runtime.s)

        current_state_idx = runtime.state_encoder(runtime.s)
        allowed_actions = runtime._get_allowed_action_indices(runtime.s)
        action_table_idx = agent.select_action(current_state_idx, allowed_actions=allowed_actions)
        get_last_action_info = getattr(agent, "get_last_action_info", None)
        action_info = get_last_action_info() if callable(get_last_action_info) else {}
        real_action_idx = runtime._resolve_action_id(runtime.s, action_table_idx)
        previous_cost = runtime.s.fitness
        previous_d_inf = runtime.s.current_d_inf
        previous_best_feasible = runtime.best_feasible_cost

        candidate = runtime.generate_candidate_by_action(runtime.s, real_action_idx)
        candidate_is_infeasible = (
            int(real_action_idx) in {4, 5}
            and not bool(getattr(candidate, "current_is_feasible", False))
        )
        current_true_cost = float(getattr(runtime.s, "true_fitness", previous_cost))
        candidate_true_cost = float(getattr(candidate, "true_fitness", getattr(candidate, "fitness", np.inf)))
        runtime._record_action_selection(
            real_action_idx,
            previous_cost,
            candidate.fitness,
            phase="main",
            selection_mode=action_info.get("selection_mode"),
            epsilon_value=action_info.get("epsilon"),
            allowed_count=action_info.get("allowed_count"),
        )
        accept, prob, _, _ = runtime._accept_candidate(runtime.s.fitness, candidate.fitness)
        if candidate_is_infeasible:
            record_infeasible_acceptance_diag = getattr(runtime, "_record_infeasible_acceptance_diag", None)
            if callable(record_infeasible_acceptance_diag):
                record_infeasible_acceptance_diag(
                    real_action_idx,
                    accept,
                    prob,
                    runtime.s.fitness,
                    candidate.fitness,
                    current_true_cost,
                    candidate_true_cost,
                    int(getattr(candidate, "current_d_inf", 0)),
                    float(getattr(candidate, "violation_sum", 0.0)),
                )
        runtime.prob_history.append(prob)
        runtime._record_acceptance(accept)

        archive_improved = False
        step_improved = False
        accepted_improved = False
        trigger_local_search = False
        run_topk_guided = False
        guided_topk = runtime.topk_guided_topk
        guided_max_iters = runtime.topk_guided_max_iters
        local_search_improved = False

        if accept:
            runtime.s = candidate
            runtime.current_energy = runtime.s.fitness
            if runtime.s.current_is_feasible:
                archive_improved = bool(runtime._observe_feasible_state(runtime.s))
            else:
                runtime._observe_archive_candidate(runtime.s)
                archive_improved = False
            if archive_improved:
                runtime._record_action_global_best(real_action_idx, phase="main")
            accepted_improved = bool(
                np.isfinite(previous_cost)
                and np.isfinite(runtime.s.fitness)
                and runtime.s.fitness < previous_cost
            )
            runtime._record_action_acceptance(
                real_action_idx,
                previous_cost,
                runtime.s.fitness,
                improved=accepted_improved,
                phase="main",
            )
            step_improved = accepted_improved
            trigger_local_search = bool(
                runtime.local_search_on_any_feasible_accept
                and runtime._should_trigger_local_search(
                    runtime.s,
                    real_action_idx,
                    accepted_improved,
                    previous_d_inf,
                )
            )
            if trigger_local_search:
                run_topk_guided, guided_topk, guided_max_iters = runtime._should_run_topk_guided_search(
                    runtime.s,
                    real_action_idx,
                )
            runtime._update_histogram(runtime.s.fitness)
        else:
            runtime._update_histogram(runtime.s.fitness)

        immediate_transition_cost = float(runtime.s.fitness)
        immediate_transition_d_inf = int(runtime.s.current_d_inf)
        if trigger_local_search:
            before_local_search_cost = float(runtime.s.fitness)
            runtime.s = runtime._greedy_local_search(
                runtime.s,
                enable_guided=run_topk_guided,
                guided_topk=guided_topk,
                guided_max_iters=guided_max_iters,
            )
            runtime.current_energy = runtime.s.fitness
            local_search_improved = bool(
                runtime.s.current_is_feasible
                and np.isfinite(before_local_search_cost)
                and np.isfinite(runtime.s.fitness)
                and runtime.s.fitness + 1e-9 < before_local_search_cost
            )
            if local_search_improved:
                runtime._update_histogram(runtime.s.fitness)

        rl_next_state_idx = runtime.state_encoder(runtime.s)
        rl_next_d_inf = runtime.s.current_d_inf
        rl_next_cost = runtime.s.fitness
        record_post_action_mode = getattr(runtime, "_record_post_action_mode", None)
        if callable(record_post_action_mode):
            record_post_action_mode(runtime.s)
        rl_allowed_next_actions = runtime._get_allowed_action_indices(runtime.s)
        step_improved = bool(
            (
                np.isfinite(previous_cost)
                and np.isfinite(rl_next_cost)
                and rl_next_cost + 1e-9 < previous_cost
            )
            or rl_next_d_inf < previous_d_inf
        )
        reward_components = runtime._compute_reward_components(
            previous_cost,
            immediate_transition_cost,
            rl_next_cost,
            previous_d_inf,
            immediate_transition_d_inf,
            rl_next_d_inf,
            previous_best_feasible,
            accept,
        )
        reward = runtime._compose_training_reward(reward_components)
        done_flag = step_idx == (runtime.t_max - 1)
        agent.update_Q(
            current_state_idx,
            action_table_idx,
            reward,
            rl_next_state_idx,
            done=done_flag,
            allowed_next_actions=rl_allowed_next_actions,
        )

        if step_improved:
            runtime.no_improve_steps = 0
        else:
            runtime.no_improve_steps += 1

        gbest_recomputed = bool(
            np.isfinite(runtime.best_feasible_cost)
            and runtime.best_feasible_cost < previous_best_feasible
        )
        if gbest_recomputed and not archive_improved:
            runtime._record_action_global_best(real_action_idx, phase="main")
        if gbest_recomputed:
            runtime._refresh_temperature_floor(allow_raise=False)
            runtime.fast_time, _ = runtime._elite_intensification(runtime.fast_time)
        runtime._record_action_final_outcome(
            real_action_idx,
            previous_cost,
            immediate_transition_cost,
            rl_next_cost,
            improved=step_improved,
            local_search_triggered=trigger_local_search,
            local_search_improved=local_search_improved,
            immediate_reward=reward_components["immediate_reward"],
            final_reward=reward_components["final_reward"],
            phase="main",
        )

        return DQNTrainingStepResult(
            action_table_idx=int(action_table_idx),
            real_action_idx=int(real_action_idx),
            reward=float(reward),
            step_improved=bool(step_improved),
            archive_improved=bool(archive_improved),
            gbest_recomputed=bool(gbest_recomputed),
            trigger_local_search=bool(trigger_local_search),
            local_search_improved=bool(local_search_improved),
        )


class DQNProgramRunner:
    """DQN 主训练链路的运行编排器。"""

    def run_training_loop(self, runtime):
        start_time = datetime.datetime.now()
        fast_time = start_time
        global_step = 0
        total_steps = max(1, runtime.G * runtime.t_max)
        agent_mode = os.getenv("ELP_RL_AGENT", "dqn").strip().lower()
        initial_state_vector = np.asarray(runtime.state_encoder(runtime.s), dtype=np.float32).reshape(-1)
        state_dim = int(initial_state_vector.size) if initial_state_vector.size > 0 else 8
        runtime.last_rl_agent_mode = agent_mode
        runtime.last_state_dim = state_dim
        if agent_mode == "qlearning":
            # 复用统一的 epsilon 配置，便于在消融实验中把 qlearning 设置为纯随机策略。
            agent = StandardQLearningAgent(
                s_dim=state_dim,
                a_dim=int(getattr(runtime, "rl_action_dim", len(runtime.valid_actions))),
                epsilon=runtime.dqn_epsilon_start,
                epsilon_min=runtime.dqn_epsilon_min,
                epsilon_decay=runtime.dqn_epsilon_decay,
            )
        else:
            agent = StandardDQNAgent(
                s_dim=state_dim,
                a_dim=int(getattr(runtime, "rl_action_dim", len(runtime.valid_actions))),
                epsilon=runtime.dqn_epsilon_start,
                epsilon_min=runtime.dqn_epsilon_min,
                epsilon_decay=runtime.dqn_epsilon_decay,
                gamma=runtime.dqn_gamma,
                lr=runtime.dqn_lr,
                batch_size=runtime.dqn_batch_size,
                replay_capacity=runtime.dqn_replay_capacity,
                warmup_steps=runtime.dqn_warmup_steps,
                update_every=runtime.dqn_update_every,
                target_update_every=runtime.dqn_target_update_every,
                idqn_k=runtime.dqn_idqn_k,
                embedding_dim=runtime.dqn_embedding_dim,
                hidden_dim=runtime.dqn_hidden_dim,
                grad_clip=runtime.dqn_grad_clip,
                step_epsilon_schedule=runtime.dqn_step_epsilon_schedule,
                epsilon_warmup_steps=runtime.dqn_epsilon_schedule_warmup_steps,
                epsilon_decay_steps=int(total_steps * runtime.dqn_epsilon_schedule_decay_ratio),
            )
        if not runtime._bootstrap_until_first_feasible():
            runtime.logger.warning("Failed to seed a feasible archive before ELP search.")
        if np.isfinite(runtime.best_feasible_cost):
            runtime.best_energy = runtime.best_feasible_cost
            runtime.current_energy = runtime.s.fitness

        next_progress_marker_idx = 0
        for episode in range(runtime.G):
            if runtime.worst_feasible_cost is None and not runtime._bootstrap_until_first_feasible(
                max_attempts=max(200, 2 * runtime.t_max)
            ):
                runtime.logger.warning(f"Episode {episode}: feasible archive still unavailable.")
                continue
            episode_best_before = runtime.best_feasible_cost
            runtime._prepare_episode_start(episode)
            for step_idx in range(runtime.t_max):
                runtime.current_global_step = int(global_step)
                runtime.current_progress_ratio = float(global_step) / float(max(1, total_steps))
                runtime.fast_time = fast_time
                runtime.dqn_transition_engine.execute_training_step(runtime, agent, step_idx)
                fast_time = runtime.fast_time

                runtime.modified_energy_history.append(runtime._tilde_energy(runtime.s.fitness))
                runtime.energy_history.append(runtime.s.fitness)
                runtime.T = max(runtime.T * runtime.cooling_per_step, runtime.T_min)
                global_step += 1
                while (
                    next_progress_marker_idx < len(runtime.progress_markers)
                    and (global_step / total_steps) >= runtime.progress_markers[next_progress_marker_idx]
                ):
                    runtime._log_training_progress(runtime.progress_markers[next_progress_marker_idx], start_time)
                    next_progress_marker_idx += 1

                post_step_result = runtime.dqn_search_controller.run_post_step_controls(
                    runtime,
                    global_step,
                    total_steps,
                    fast_time,
                )
                fast_time = post_step_result.fast_time

            finalize_episode = getattr(agent, "finalize_episode", None)
            if callable(finalize_episode):
                finalize_episode()

            if np.isfinite(runtime.best_feasible_cost) and runtime.best_feasible_cost < episode_best_before:
                runtime.episodes_without_improvement = 0
            else:
                runtime.episodes_without_improvement += 1
            agent.decay_epsilon()

        while next_progress_marker_idx < len(runtime.progress_markers):
            runtime._log_training_progress(runtime.progress_markers[next_progress_marker_idx], start_time)
            next_progress_marker_idx += 1

        end_time = datetime.datetime.now()
        best_solution = runtime.best_feasible_solution if runtime.best_feasible_solution is not None else copy.deepcopy(runtime.s)
        is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        best_energy = float(runtime.best_feasible_cost if np.isfinite(runtime.best_feasible_cost) else best_solution.fitness)
        return (
            runtime.G * runtime.t_max,
            is_valid,
            best_solution,
            best_energy,
            start_time,
            end_time,
            fast_time,
        )

    def get_initial_solution_energy(self, env_obj, fbs_util):
        fitness = float(getattr(env_obj, "fitness", np.inf))
        if np.isfinite(fitness):
            return fitness

        evaluator = getattr(env_obj, "_evaluate_current_layout", None)
        if callable(evaluator):
            evaluator(snapshot_best=True)
            fitness = float(getattr(env_obj, "fitness", np.inf))
            if np.isfinite(fitness):
                return fitness

        metrics = fbs_util.evaluate_layout(
            env_obj.fbs_model,
            env_obj.areas,
            env_obj.H,
            env_obj.F,
            env_obj.aspect_limits,
            v_ref=getattr(env_obj, "current_v_ref", getattr(env_obj, "current_v_worst", None)),
            k_penalty=getattr(env_obj, "k_penalty", 1.0),
            tau=getattr(env_obj, "true_cost_tau", 0.2),
            alpha=getattr(env_obj, "true_cost_alpha", 1.0),
            beta=getattr(env_obj, "true_cost_beta", 5.0),
            distance_metric=getattr(env_obj, "distance_metric", "manhattan"),
        )
        fitness = float(metrics["cost"])
        if np.isfinite(fitness):
            return fitness
        return float(metrics["mhc"])

    def run_script(
        self,
        *,
        elp_cls,
        runtime_config,
        logger,
        gym,
        experiments_util,
        fbs_util,
        env_flag,
        env_int,
        env_float,
        env_int_list,
        set_global_seed,
        os_module,
        copy_module,
    ):
        runtime_config_tag = str(
            getattr(runtime_config, "ELP_RUNTIME_CONFIG_NAME", "default_runtime")
        ).strip()
        exp_instance = os_module.getenv("ELP_EXP_INSTANCE", "Du62")
        exp_algorithm = os_module.getenv(
            "ELP_EXP_ALGORITHM",
            f"ELP_RL_Standard_{runtime_config_tag}",
        )
        exp_remark_default = (
            "ELP+DQN(main:0/1/2/3/9/10/11/14/15; elite:10/11/14/0; heavy_two_stage:9/10/11/14/15; "
            "state:restructured_facility+search_context; reward:train=0.8immediate+0.2final; "
            "dqn:1200warmup+128hidden; tracked=immediate+final)"
        )
        exp_remark = os_module.getenv("ELP_EXP_REMARK", exp_remark_default)
        exp_number = env_int("ELP_EXP_NUMBER", 12)

        is_exp = env_flag("ELP_IS_EXP", True)
        print_telemetry = env_flag("ELP_PRINT_TELEMETRY", False)
        save_telemetry_csv = env_flag("ELP_SAVE_TELEMETRY_CSV", True)
        save_experiment_result = env_flag("ELP_SAVE_EXPERIMENT_RESULT", True)

        G = env_int("ELP_G", 600)
        t_max = env_int("ELP_T_MAX", 240)
        T_initial = env_float("ELP_T_INITIAL", 3000.0)
        k_hist = env_float("ELP_K_HIST", 5.0)

        fixed_seeds = env_int_list("ELP_FIXED_SEEDS")
        base_seed = env_int("ELP_BASE_SEED", 20260328)
        if fixed_seeds:
            exp_number = len(fixed_seeds)

        def _seed_for_run(run_index):
            if fixed_seeds:
                return int(fixed_seeds[run_index])
            return int(base_seed + run_index)

        if is_exp:
            for i in range(exp_number):
                run_seed = _seed_for_run(i)
                set_global_seed(run_seed)
                logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
                logger.info(f"Experiment seed: {run_seed}")
                try:
                    env = gym.make("FbsEnv-v0", instance=exp_instance)
                    try:
                        env.reset(seed=run_seed)
                    except TypeError:
                        env.reset()
                    except Exception:
                        env.reset()
                    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
                    initial_gbest = copy_module.deepcopy(base_env)
                    logger.info(f"Initial solution energy: {self.get_initial_solution_energy(base_env, fbs_util)}")
                    elp_solver = elp_cls(
                        env=base_env,
                        gbest=initial_gbest,
                        T=T_initial,
                        G=G,
                        t_max=t_max,
                        k=k_hist,
                    )
                    total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()
                    logger.info(f"Experiment {i + 1} complete | best energy: {best_energy}")
                    if print_telemetry:
                        for telemetry_line in elp_solver.format_action_telemetry():
                            logger.info(f"Telemetry | {telemetry_line}")
                    if save_telemetry_csv:
                        experiments_util.save_action_telemetry_rows(
                            elp_solver.build_action_telemetry_rows(
                                exp_instance=exp_instance,
                                exp_algorithm=exp_algorithm,
                                exp_remark=exp_remark,
                                run_seed=run_seed,
                                run_index=i + 1,
                                is_valid=is_valid,
                                best_energy=best_energy,
                                start_time=start,
                                end_time=end,
                                fast_time=fast,
                            )
                        )
                    if save_experiment_result:
                        experiments_util.save_experiment_result(
                            exp_instance=exp_instance,
                            exp_algorithm=exp_algorithm,
                            exp_iterations=total_iter,
                            exp_solution=best_sol.fbs_model.array_2d,
                            exp_fitness=best_energy,
                            exp_start_time=start,
                            exp_fast_time=fast,
                            exp_end_time=end,
                            exp_is_valid_aspect_ratio=is_valid,
                            exp_remark=exp_remark,
                            exp_gbest_updates=elp_solver.gbest_update_count,
                        )
                except Exception as exc:
                    logger.exception(f"Experiment {i + 1} failed: {exc}")
        else:
            run_seed = _seed_for_run(0)
            set_global_seed(run_seed)
            logger.info(f"Single-run seed: {run_seed}")
            env = gym.make("FbsEnv-v0", instance=exp_instance)
            try:
                env.reset(seed=run_seed)
            except TypeError:
                env.reset()
            except Exception:
                env.reset()
            base_env = env.unwrapped if hasattr(env, "unwrapped") else env
            initial_gbest = copy_module.deepcopy(base_env)
            logger.info(f"Initial solution energy: {self.get_initial_solution_energy(base_env, fbs_util)}")
            elp_solver = elp_cls(
                env=base_env,
                gbest=initial_gbest,
                T=T_initial,
                G=G,
                t_max=t_max,
                k=k_hist,
            )
            total_iter, is_valid, best_sol, best_energy, start, end, fast = elp_solver.run()
            print(f"Single run complete | best energy: {best_energy}")
            if print_telemetry:
                for telemetry_line in elp_solver.format_action_telemetry():
                    print(f"Telemetry | {telemetry_line}")
            if save_telemetry_csv:
                experiments_util.save_action_telemetry_rows(
                    elp_solver.build_action_telemetry_rows(
                        exp_instance=exp_instance,
                        exp_algorithm=exp_algorithm,
                        exp_remark=exp_remark,
                        run_seed=run_seed,
                        run_index=1,
                        is_valid=is_valid,
                        best_energy=best_energy,
                        start_time=start,
                        end_time=end,
                        fast_time=fast,
                    )
                )
            if save_experiment_result:
                experiments_util.save_experiment_result(
                    exp_instance=exp_instance,
                    exp_algorithm=exp_algorithm,
                    exp_iterations=total_iter,
                    exp_solution=best_sol.fbs_model.array_2d,
                    exp_fitness=best_energy,
                    exp_start_time=start,
                    exp_fast_time=fast,
                    exp_end_time=end,
                    exp_is_valid_aspect_ratio=is_valid,
                    exp_remark=exp_remark,
                    exp_gbest_updates=elp_solver.gbest_update_count,
                )
