from typing import Optional

from eve_rl import Runner


class WandbRunner(Runner):
    def __init__(self, *args, wandb_run=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wandb_run = wandb_run

    def heatup(self, steps: int):
        result = super().heatup(steps)
        print(
            "Heatup complete: "
            f"steps={self.step_counter.heatup}, "
            f"episodes={self.episode_counter.heatup}"
        )
        metrics = {
            "train/heatup_steps": self.step_counter.heatup,
            "train/heatup_episodes": self.episode_counter.heatup,
        }
        metrics.update(_episode_metrics(result, "heatup"))
        self._log_wandb(metrics)
        return result

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: Optional[int] = None,
        explore_steps_limit: Optional[int] = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        explore_result, update_result = [], []
        while self.step_counter.exploration < explore_steps_limit:
            update_steps = (
                self.step_counter.exploration * update_steps_per_explore_step
                - self.step_counter.update
            )
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )
            if result is not None:
                explore_result, update_result = result
            self._log_train_progress(update_result, explore_result)

        print(
            "Explore/update complete: "
            f"exploration_steps={self.step_counter.exploration}, "
            f"exploration_episodes={self.episode_counter.exploration}, "
            f"update_steps={self.step_counter.update}"
        )
        return explore_result, update_result

    def _log_train_progress(self, update_result, explore_result=None):
        metrics = {
            "train/exploration_steps": self.step_counter.exploration,
            "train/exploration_episodes": self.episode_counter.exploration,
            "train/update_steps": self.step_counter.update,
        }
        metrics.update(_loss_metrics(update_result))
        metrics.update(_episode_metrics(explore_result, "explore"))
        self._log_wandb(metrics)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[list] = None
    ):
        if seeds is not None:
            print(f"Starting evaluation: seed_count={len(seeds)}")
        elif episodes is not None:
            print(f"Starting evaluation: episodes={episodes}")
        else:
            print("Starting evaluation")
        quality, reward = super().eval(episodes=episodes, seeds=seeds)
        print(
            "Evaluation complete: "
            f"quality={quality:.3f}, "
            f"reward={reward:.3f}, "
            f"best_quality={self._results.get('best quality')}"
        )
        metrics = {
            "eval/quality": quality,
            "eval/reward": reward,
            "eval/best_quality": self._results.get("best quality"),
            "eval/best_exploration_steps": self._results.get("best explore steps"),
            "eval/steps": self.step_counter.evaluation,
            "eval/episodes": self.episode_counter.evaluation,
            "train/exploration_steps": self.step_counter.exploration,
            "train/exploration_episodes": self.episode_counter.exploration,
            "train/update_steps": self.step_counter.update,
            "train/heatup_steps": self.step_counter.heatup,
        }
        for info_result in self.info_results:
            metric_name = _metric_name(info_result)
            value = self._results.get(info_result)
            metrics[f"env/{metric_name}"] = value
            metrics[f"eval/env/{metric_name}"] = value
        self._log_wandb(metrics)
        if self.wandb_run is not None:
            self.wandb_run.summary["best_quality"] = self._results.get("best quality")
            self.wandb_run.summary["best_exploration_steps"] = self._results.get(
                "best explore steps"
            )
        return quality, reward

    def _log_wandb(self, metrics) -> None:
        if self.wandb_run is None:
            return
        clean_metrics = {}
        for key, value in metrics.items():
            number = _to_number(value)
            if number is not None:
                clean_metrics[key] = number
        if clean_metrics:
            self.wandb_run.log(clean_metrics, step=self.step_counter.exploration)


def _loss_metrics(update_result):
    if not update_result:
        return {}
    q1_losses = []
    q2_losses = []
    policy_losses = []
    for result in update_result:
        if result is None or len(result) < 3:
            continue
        q1_losses.append(float(result[0]))
        q2_losses.append(float(result[1]))
        policy_losses.append(float(result[2]))
    if not q1_losses:
        return {}
    return {
        "train/q1_loss": sum(q1_losses) / len(q1_losses),
        "train/q2_loss": sum(q2_losses) / len(q2_losses),
        "train/policy_loss": sum(policy_losses) / len(policy_losses),
    }


def _episode_metrics(episodes, prefix: str):
    if not episodes:
        return {}
    episode_lengths = [len(episode) for episode in episodes]
    episode_rewards = [float(episode.episode_reward) for episode in episodes]
    terminals = [
        float(bool(episode.terminals[-1])) for episode in episodes if episode.terminals
    ]
    truncations = [
        float(bool(episode.truncations[-1]))
        for episode in episodes
        if episode.truncations
    ]
    metrics = {
        f"{prefix}/episode_count": len(episodes),
        f"{prefix}/episode_length_mean": sum(episode_lengths) / len(episode_lengths),
        f"{prefix}/episode_reward_mean": sum(episode_rewards) / len(episode_rewards),
        f"{prefix}/terminal_rate": _mean(terminals),
        f"{prefix}/truncation_rate": _mean(truncations),
    }

    final_infos = [episode.infos[-1] for episode in episodes if episode.infos]
    info_values = {}
    for info in final_infos:
        for key, value in info.items():
            number = _to_number(value)
            if number is None:
                continue
            info_values.setdefault(_metric_name(key), []).append(number)
    for key, values in info_values.items():
        metrics[f"{prefix}/env/{key}"] = sum(values) / len(values)
    return metrics


def _mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def _metric_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _to_number(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
