import argparse
import os
import logging
import torch.multiprocessing as mp
import torch
import optuna
from util.agent import BenchAgentSynchron
from util.env import BenchEnv
from util.optunapruner import CombinationPruner, StagnatingPruner
from util.schedule_args import add_schedule_args, schedule_from_args
from util.util import get_result_checkpoint_config_and_log_path
from util.wandb_runner import WandbRunner
from util.wandb_tracking import add_wandb_args, finish_wandb, init_wandb
from eve_bench import ArchVariety


RESULTS_FOLDER = (
    os.getcwd() + "/results/eve_paper/neurovascular/aorta/gw_only/typeI_hyperparam_opti"
)

EVAL_SEEDS = "1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,44,47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,95,97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,129,130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,152,154,155,156,158,159,161,162,167,168,171,175"
EVAL_SEEDS = EVAL_SEEDS.split(",")
EVAL_SEEDS = [int(seed) for seed in EVAL_SEEDS]

HEATUP_STEPS = 5e5
TRAINING_STEPS = 1e7
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = 5e5

# HEATUP_STEPS = 5e3
# TRAINING_STEPS = 1e7
# CONSECUTIVE_EXPLORE_EPISODES = 10
# EXPLORE_STEPS_BTW_EVAL = 2.5e3
# EVAL_SEEDS = list(range(20))
# RESULTS_FOLDER = os.getcwd() + "/results/test"


GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20


LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.INFO


if __name__ == "__main__":

    def objective(trial: optuna.trial.Trial):

        (
            results_file,
            checkpoint_folder,
            config_folder,
            log_file,
        ) = get_result_checkpoint_config_and_log_path(
            all_results_folder=RESULTS_FOLDER, name=trial_name
        )

        logging.basicConfig(
            filename=log_file,
            level=DEBUG_LEVEL,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
        lr = trial.suggest_float("lr", 8e-5, 2e-3, log=True)
        n_hidden_layer = trial.suggest_int("n_hidden_layer", 2, 4)
        hidden_layer_nodes = trial.suggest_int("hidden_layer_nodes", 300, 900, step=100)
        hidden_layers = [hidden_layer_nodes] * n_hidden_layer
        embedder_nodes = trial.suggest_int("embedder_nodes", 200, 700, step=100)
        embedder_layers = trial.suggest_int("embedder_layers", 1, 2)

        custom_parameters = {
            "learning_rate": lr,
            "hidden_layers": hidden_layers,
            "embedder_nodes": embedder_nodes,
            "embedder_layers": embedder_layers,
            "HEATUP_STEPS": schedule["heatup_steps"],
            "EXPLORE_STEPS_BTW_EVAL": schedule["eval_interval"],
            "CONSECUTIVE_EXPLORE_EPISODES": schedule["explore_episodes"],
            "BATCH_SIZE": BATCH_SIZE,
            "UPDATE_PER_EXPLORE_STEP": schedule["update_per_explore_step"],
        }
        wandb_config = {
            **custom_parameters,
            "environment": "ArchVariety",
            "script": os.path.basename(__file__),
            "optimizer": "optuna",
            "trial_number": trial.number,
            "trainer_device": trainer_device,
            "worker_device": worker_device,
            "n_workers": n_worker,
            "gamma": GAMMA,
            "reward_scaling": REWARD_SCALING,
            "replay_buffer_size": REPLAY_BUFFER_SIZE,
            "consecutive_action_steps": CONSECUTIVE_ACTION_STEPS,
            "lr_end_factor": LR_END_FACTOR,
            "lr_linear_end_steps": LR_LINEAR_END_STEPS,
            "training_steps": schedule["training_steps"],
            "eval_seed_count": len(eval_seeds),
            "stochastic_eval": stochastic_eval,
        }
        wandb_run = init_wandb(
            args,
            run_name=f"{trial_name}_trial_{trial.number}",
            config=wandb_config,
            tags=["ArchVariety", "Optuna", "SAC", "RL"],
        )
        intervention = ArchVariety(episodes_between_arch_change=1)
        env_train = BenchEnv(
            intervention=intervention, mode="train", visualisation=False
        )
        intervention = ArchVariety(episodes_between_arch_change=1)
        env_eval = BenchEnv(intervention=intervention, mode="eval", visualisation=False)
        agent = BenchAgentSynchron(
            trainer_device,
            worker_device,
            lr,
            LR_END_FACTOR,
            LR_LINEAR_END_STEPS,
            hidden_layers,
            embedder_nodes,
            embedder_layers,
            GAMMA,
            BATCH_SIZE,
            REWARD_SCALING,
            REPLAY_BUFFER_SIZE,
            env_train,
            env_eval,
            CONSECUTIVE_ACTION_STEPS,
            n_worker,
            stochastic_eval,
            False,
        )

        env_train_config = os.path.join(config_folder, "env_train.yml")
        env_train.save_config(env_train_config)
        env_eval_config = os.path.join(config_folder, "env_eval.yml")
        env_eval.save_config(env_eval_config)
        infos = list(env_eval.info.info.keys())

        runner = WandbRunner(
            agent=agent,
            heatup_action_low=[-10.0, -1.0],
            heatup_action_high=[25, 3.14],
            agent_parameter_for_result_file=custom_parameters,
            checkpoint_folder=checkpoint_folder,
            results_file=results_file,
            info_results=infos,
            quality_info="success",
            wandb_run=wandb_run,
        )
        runner_config = os.path.join(config_folder, "runner.yml")
        runner.save_config(runner_config)

        quality = float("-inf")
        try:
            runner.heatup(schedule["heatup_steps"])
            next_eval_limit = schedule["eval_interval"]
            while runner.step_counter.exploration < schedule["training_steps"]:
                runner.explore_and_update(
                    schedule["explore_episodes"],
                    schedule["update_per_explore_step"],
                    explore_steps=schedule["eval_interval"],
                )
                quality, _ = runner.eval(seeds=eval_seeds)
                trial.report(quality, runner.step_counter.exploration)
                next_eval_limit += schedule["eval_interval"]

                if trial.should_prune():
                    raise optuna.TrialPruned()
                if agent.update_error:
                    break
        finally:
            agent.close()
            finish_wandb(wandb_run)
            del agent
            del intervention
            del env_eval
            del env_train
            del runner
            torch.cuda.empty_cache()
        return quality

    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="perform IJCARS23 Optuna Optimization")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=5, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, wehre the NN update is performed. ",
        choices=["cpu", "cuda:0", "cuda:1", "cuda", "mps"],
    )
    parser.add_argument(
        "-se",
        "--stochastic_eval",
        action="store_true",
        help="Runs optuna run with stochastic eval function of SAC.",
    )
    parser.add_argument(
        "-n", "--name", type=str, default="run", help="Name of the training run"
    )
    add_schedule_args(
        parser,
        heatup_steps=HEATUP_STEPS,
        training_steps=TRAINING_STEPS,
        eval_interval=EXPLORE_STEPS_BTW_EVAL,
        explore_episodes=CONSECUTIVE_EXPLORE_EPISODES,
        update_per_explore_step=UPDATE_PER_EXPLORE_STEP,
        eval_seed_count=len(EVAL_SEEDS),
    )
    add_wandb_args(parser)

    args = parser.parse_args()

    trainer_device = torch.device(args.device)
    n_worker = args.n_worker
    trial_name = args.name
    stochastic_eval = args.stochastic_eval
    worker_device = torch.device("cpu")
    schedule = schedule_from_args(args)
    eval_seeds = EVAL_SEEDS[: schedule["eval_seed_count"]]
    print(
        "Starting ArchVariety Optuna: "
        f"heatup_steps={schedule['heatup_steps']}, "
        f"training_steps={schedule['training_steps']}, "
        f"eval_interval={schedule['eval_interval']}, "
        f"explore_episodes={schedule['explore_episodes']}, "
        f"eval_seed_count={len(eval_seeds)}, "
        f"workers={n_worker}, device={trainer_device}"
    )

    pruner_median = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=TRAINING_STEPS / 5
    )
    pruner_threshold = optuna.pruners.ThresholdPruner(
        lower=0.2, n_warmup_steps=TRAINING_STEPS / 3
    )
    stagnation_prunter = StagnatingPruner(
        fluctuation_boundary=0.01,
        n_warmup_steps=TRAINING_STEPS / 4,
        n_averaged_values=10,
        n_strikes=5,
    )
    pruner = CombinationPruner(
        pruners=[pruner_median, pruner_threshold, stagnation_prunter]
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(objective, 20)
    logging.basicConfig(
        filename=RESULTS_FOLDER + "main.log",
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("main")
    log_info = f"{study.best_params = }"
    logger.info(log_info)
    param_importance = optuna.importance.get_param_importances(study)
    log_info = f"{param_importance = }"
    logger.info(log_info)
