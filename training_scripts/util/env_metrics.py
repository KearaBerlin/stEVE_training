import numpy as np

import eve
from eve.intervention.vesseltree.vesseltree import at_tree_end
from eve.util.coordtransform import tracking3d_to_vessel_cs


EPSILON = 1e-8
METRIC_KEYS = (
    "target_distance_3d",
    "target_distance_2d",
    "target_distance_delta",
    "target_distance_reduction",
    "target_distance_reduction_ratio",
    "target_distance_min",
    "target_distance_mean",
    "path_length_to_target",
    "path_length_delta",
    "path_length_reduction",
    "path_length_reduction_ratio",
    "path_length_min",
    "path_length_mean",
    "insertion_length",
    "insertion_delta",
    "insertion_fraction",
    "action_translation_abs",
    "action_rotation_abs",
    "action_translation_abs_mean",
    "action_rotation_abs_mean",
    "truncated_max_steps",
    "truncated_vessel_end",
    "truncated_sim_error",
)


class NavigationMetrics(eve.info.Info):
    """Readable progress metrics for endovascular navigation rollouts."""

    def __init__(
        self,
        intervention: eve.intervention.Intervention,
        pathfinder: eve.pathfinder.Pathfinder,
        n_max_steps: int,
    ) -> None:
        super().__init__("navigation_metrics")
        self.intervention = intervention
        self.pathfinder = pathfinder
        self.n_max_steps = n_max_steps
        self._steps = 0
        self._initial_target_distance_3d = 0.0
        self._previous_target_distance_3d = 0.0
        self._target_distance_delta = 0.0
        self._target_distance_sum = 0.0
        self._target_distance_min = 0.0
        self._initial_path_length = 0.0
        self._previous_path_length = 0.0
        self._path_length_delta = 0.0
        self._path_length_sum = 0.0
        self._path_length_min = 0.0
        self._initial_insertion_length = 0.0
        self._translation_abs_sum = 0.0
        self._rotation_abs_sum = 0.0
        self._ready = False

    @property
    def info(self):
        if not self._ready:
            return {key: 0.0 for key in METRIC_KEYS}

        target_distance_3d = self._target_distance_3d()
        target_distance_2d = self._target_distance_2d()
        path_length = self._path_length()
        insertion_length = self._insertion_length()
        insertion_fraction = self._insertion_fraction()
        translation_abs, rotation_abs = self._action_abs()
        step_count = max(self._steps, 1)
        target_distance_mean = (
            target_distance_3d
            if self._steps == 0
            else self._target_distance_sum / step_count
        )
        path_length_mean = (
            path_length if self._steps == 0 else self._path_length_sum / step_count
        )

        return {
            "target_distance_3d": target_distance_3d,
            "target_distance_2d": target_distance_2d,
            "target_distance_delta": self._target_distance_delta,
            "target_distance_reduction": self._initial_target_distance_3d
            - target_distance_3d,
            "target_distance_reduction_ratio": _safe_ratio(
                self._initial_target_distance_3d - target_distance_3d,
                self._initial_target_distance_3d,
            ),
            "target_distance_min": self._target_distance_min,
            "target_distance_mean": target_distance_mean,
            "path_length_to_target": path_length,
            "path_length_delta": self._path_length_delta,
            "path_length_reduction": self._initial_path_length - path_length,
            "path_length_reduction_ratio": _safe_ratio(
                self._initial_path_length - path_length,
                self._initial_path_length,
            ),
            "path_length_min": self._path_length_min,
            "path_length_mean": path_length_mean,
            "insertion_length": insertion_length,
            "insertion_delta": insertion_length - self._initial_insertion_length,
            "insertion_fraction": insertion_fraction,
            "action_translation_abs": translation_abs,
            "action_rotation_abs": rotation_abs,
            "action_translation_abs_mean": self._translation_abs_sum / step_count,
            "action_rotation_abs_mean": self._rotation_abs_sum / step_count,
            "truncated_max_steps": float(self._steps >= self.n_max_steps),
            "truncated_vessel_end": float(self._at_vessel_end()),
            "truncated_sim_error": float(self.intervention.simulation.simulation_error),
        }

    def step(self) -> None:
        self._steps += 1
        target_distance = self._target_distance_3d()
        path_length = self._path_length()
        translation_abs, rotation_abs = self._action_abs()

        self._target_distance_delta = self._previous_target_distance_3d - target_distance
        self._target_distance_sum += target_distance
        self._target_distance_min = min(self._target_distance_min, target_distance)
        self._previous_target_distance_3d = target_distance

        self._path_length_delta = self._previous_path_length - path_length
        self._path_length_sum += path_length
        self._path_length_min = min(self._path_length_min, path_length)
        self._previous_path_length = path_length

        self._translation_abs_sum += translation_abs
        self._rotation_abs_sum += rotation_abs

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr
        self._steps = 0
        self._initial_target_distance_3d = self._target_distance_3d()
        self._previous_target_distance_3d = self._initial_target_distance_3d
        self._target_distance_delta = 0.0
        self._target_distance_sum = 0.0
        self._target_distance_min = self._initial_target_distance_3d

        self._initial_path_length = self._path_length()
        self._previous_path_length = self._initial_path_length
        self._path_length_delta = 0.0
        self._path_length_sum = 0.0
        self._path_length_min = self._initial_path_length

        self._initial_insertion_length = self._insertion_length()
        self._translation_abs_sum = 0.0
        self._rotation_abs_sum = 0.0
        self._ready = True

    def _target_distance_3d(self) -> float:
        tip = np.asarray(self.intervention.fluoroscopy.tracking3d[0], dtype=np.float64)
        target = np.asarray(self.intervention.target.coordinates3d, dtype=np.float64)
        return float(np.linalg.norm(tip - target))

    def _target_distance_2d(self) -> float:
        tip = np.asarray(self.intervention.fluoroscopy.tracking2d[0], dtype=np.float64)
        target = np.asarray(self.intervention.target.coordinates2d, dtype=np.float64)
        return float(np.linalg.norm(tip - target))

    def _path_length(self) -> float:
        return float(self.pathfinder.path_length)

    def _insertion_length(self) -> float:
        lengths = np.asarray(self.intervention.device_lengths_inserted, dtype=np.float64)
        if lengths.size == 0:
            return 0.0
        return float(np.max(lengths))

    def _insertion_fraction(self) -> float:
        lengths = np.asarray(self.intervention.device_lengths_inserted, dtype=np.float64)
        max_lengths = np.asarray(
            self.intervention.device_lengths_maximum, dtype=np.float64
        )
        if lengths.size == 0 or max_lengths.size == 0:
            return 0.0
        return _safe_ratio(float(np.max(lengths)), float(np.max(max_lengths)))

    def _action_abs(self):
        action = np.asarray(self.intervention.last_action, dtype=np.float64)
        if action.size == 0:
            return 0.0, 0.0
        action = action.reshape(-1, 2)
        translation_abs = float(np.mean(np.abs(action[:, 0])))
        rotation_abs = float(np.mean(np.abs(action[:, 1])))
        return translation_abs, rotation_abs

    def _at_vessel_end(self) -> bool:
        fluoroscopy = self.intervention.fluoroscopy
        tip = fluoroscopy.tracking3d[0]
        tip_vessel_cs = tracking3d_to_vessel_cs(
            tip,
            fluoroscopy.image_rot_zx,
            fluoroscopy.image_center,
        )
        return bool(at_tree_end(tip_vessel_cs, self.intervention.vessel_tree))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < EPSILON:
        return 0.0
    return float(numerator / denominator)
