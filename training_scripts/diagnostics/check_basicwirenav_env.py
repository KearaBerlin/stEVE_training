#!/usr/bin/env python3
"""Quick BasicWireNav environment sanity check and GIF recorder."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
for path in (
    REPO_ROOT / "eve",
    REPO_ROOT / "eve_bench",
    REPO_ROOT / "training_scripts",
):
    sys.path.insert(0, str(path))

from eve_bench import BasicWireNav  # noqa: E402
from util.env import BenchEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short BasicWireNav rollouts and check env health."
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--policy",
        choices=["zero", "forward", "rotate", "wiggle", "random", "heatup", "naive_target"],
        default="naive_target",
    )
    parser.add_argument(
        "--translation_fraction",
        type=float,
        default=0.35,
        help="Fraction of max forward translation used by simple policies.",
    )
    parser.add_argument(
        "--rotation_fraction",
        type=float,
        default=0.35,
        help="Fraction of max rotation used by rotate/wiggle policies.",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Optional output GIF path, for example diagnostics/basicwirenav_check.gif.",
    )
    parser.add_argument("--gif_every", type=int, default=2)
    parser.add_argument("--gif_max_frames", type=int, default=200)
    parser.add_argument("--gif_fps", type=float, default=8.0)
    parser.add_argument(
        "--mesh_gif",
        type=Path,
        default=None,
        help="Optional SOFA/OpenGL mesh GIF path.",
    )
    parser.add_argument("--mesh_gif_every", type=int, default=2)
    parser.add_argument("--mesh_gif_max_frames", type=int, default=200)
    parser.add_argument("--mesh_gif_fps", type=float, default=8.0)
    parser.add_argument(
        "--sofa_window",
        action="store_true",
        help="Open the SOFA/Pygame visualizer and call render() during rollout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mesh_gif is not None:
        args.sofa_window = True
    rng = np.random.default_rng(args.seed)
    issues: list[str] = []
    frames = []
    mesh_frames = []

    print(f"SOFA_ROOT={os.environ.get('SOFA_ROOT', '<not set>')}")
    if not _check_sofa_import():
        return 2

    env = BenchEnv(
        intervention=BasicWireNav(),
        mode=args.mode,
        visualisation=args.sofa_window,
        n_max_steps=args.max_steps,
    )
    print(
        "Environment: "
        f"BasicWireNav mode={args.mode}, max_steps={args.max_steps}, "
        f"policy={args.policy}, sofa_window={args.sofa_window}"
    )
    print(f"Action space shape={env.action_space.shape}")
    print(f"Action low={env.action_space.low.tolist()}")
    print(f"Action high={env.action_space.high.tolist()}")
    print(f"Observation keys={list(env.observation_space.spaces.keys())}")

    try:
        for episode_idx in range(args.episodes):
            seed = args.seed + episode_idx
            obs, info = env.reset(seed=seed)
            if args.sofa_window:
                image = env.render()
                _maybe_add_mesh_frame(mesh_frames, args, image, 0, force=True)

            episode_reward = 0.0
            terminal = False
            truncated = False
            step_count = 0
            action_history = []
            last_info = info
            issues.extend(_nonfinite_issues(obs, f"episode_{episode_idx}.reset_obs"))
            issues.extend(_nonfinite_issues(info, f"episode_{episode_idx}.reset_info"))
            _maybe_add_frame(frames, args, obs, 0, 0.0, terminal, truncated, None)

            start_time = time.perf_counter()
            for step_idx in range(1, args.max_steps + 1):
                action = _policy_action(env, obs, args, rng, step_idx)
                action_history.append(action.copy())
                obs, reward, terminal, truncated, info = env.step(action)
                if args.sofa_window:
                    image = env.render()
                    _maybe_add_mesh_frame(
                        mesh_frames,
                        args,
                        image,
                        step_idx,
                        force=terminal or truncated,
                    )

                step_count = step_idx
                episode_reward += float(reward)
                last_info = info
                issues.extend(_nonfinite_issues(obs, f"episode_{episode_idx}.obs[{step_idx}]"))
                issues.extend(
                    _nonfinite_issues(reward, f"episode_{episode_idx}.reward[{step_idx}]")
                )
                issues.extend(_nonfinite_issues(info, f"episode_{episode_idx}.info[{step_idx}]"))
                _maybe_add_frame(
                    frames, args, obs, step_idx, reward, terminal, truncated, action
                )

                if terminal or truncated:
                    break

            elapsed = max(time.perf_counter() - start_time, 1e-9)
            if not terminal and not truncated:
                issues.append(
                    f"episode_{episode_idx} did not report terminal/truncated by "
                    f"{args.max_steps} steps"
                )
            _print_episode_summary(
                episode_idx,
                seed,
                step_count,
                episode_reward,
                terminal,
                truncated,
                last_info,
                action_history,
                elapsed,
            )
    finally:
        try:
            env.close()
        except Exception as exc:  # noqa: BLE001
            issues.append(f"env.close() raised {type(exc).__name__}: {exc}")

    if args.gif is not None:
        _save_gif(frames, args.gif, args.gif_fps)
    if args.mesh_gif is not None:
        _save_gif(mesh_frames, args.mesh_gif, args.mesh_gif_fps)

    if issues:
        print("\nCHECK FAILED")
        for issue in issues[:25]:
            print(f"- {issue}")
        if len(issues) > 25:
            print(f"- ... {len(issues) - 25} more issue(s)")
        return 1

    print("\nCHECK PASSED")
    print(
        "No non-finite obs/reward/info values were found, and every episode "
        "ended by terminal or truncation within max_steps."
    )
    return 0


def _check_sofa_import() -> bool:
    try:
        import Sofa  # noqa: F401
        import SofaRuntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"SOFA import failed: {type(exc).__name__}: {exc}")
        print("Run this first:")
        print("  export SOFA_ROOT=$HOME/sofa_stEVE/install")
        print(
            "  export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH"
        )
        print("  python -c \"import Sofa; import SofaRuntime; print('SOFA OK')\"")
        return False
    print("SOFA import OK")
    return True


def _policy_action(
    env: BenchEnv,
    obs: dict[str, Any],
    args: argparse.Namespace,
    rng: np.random.Generator,
    step_idx: int,
) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    action = np.zeros_like(high, dtype=np.float32)
    max_translation = float(high[0, 0])
    max_rotation = float(high[0, 1])

    if args.policy == "zero":
        return action
    if args.policy == "random":
        return rng.uniform(low, high).astype(np.float32)
    if args.policy == "heatup":
        heatup_low = np.array([[-10.0, -1.0]], dtype=np.float32)
        heatup_high = np.array([[25.0, 3.14]], dtype=np.float32)
        return np.clip(rng.uniform(heatup_low, heatup_high), low, high).astype(np.float32)

    action[0, 0] = args.translation_fraction * max_translation
    if args.policy == "forward":
        return np.clip(action, low, high)
    if args.policy == "rotate":
        action[0, 1] = args.rotation_fraction * max_rotation
        return np.clip(action, low, high)
    if args.policy == "wiggle":
        action[0, 1] = (
            math.sin(step_idx * 0.25) * args.rotation_fraction * max_rotation
        )
        return np.clip(action, low, high)

    tracking = _current_tracking(obs)
    target = np.asarray(obs["target"], dtype=np.float32).reshape(-1)[:2]
    tip = tracking[0]
    heading = tip - tracking[min(1, len(tracking) - 1)]
    to_target = target - tip
    heading_norm = float(np.linalg.norm(heading))
    target_norm = float(np.linalg.norm(to_target))
    if heading_norm > 1e-6 and target_norm > 1e-6:
        heading = heading / heading_norm
        to_target = to_target / target_norm
        signed = float(heading[0] * to_target[1] - heading[1] * to_target[0])
        dot = float(np.clip(np.dot(heading, to_target), -1.0, 1.0))
        angle = math.atan2(signed, dot)
        action[0, 1] = np.clip(angle / (math.pi / 2), -1.0, 1.0) * 0.45 * max_rotation
        action[0, 0] = (0.12 if abs(angle) > 1.0 else 0.35) * max_translation
    return np.clip(action, low, high)


def _current_tracking(obs: dict[str, Any]) -> np.ndarray:
    tracking = np.asarray(obs["tracking"], dtype=np.float32)
    if tracking.ndim == 3:
        return tracking[0]
    return tracking.reshape(-1, 2)


def _nonfinite_issues(value: Any, name: str) -> list[str]:
    if isinstance(value, dict):
        issues = []
        for key, item in value.items():
            issues.extend(_nonfinite_issues(item, f"{name}.{key}"))
        return issues
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return []
    if not np.all(np.isfinite(array)):
        return [f"{name} has non-finite value(s): shape={array.shape}"]
    return []


def _print_episode_summary(
    episode_idx: int,
    seed: int,
    step_count: int,
    episode_reward: float,
    terminal: bool,
    truncated: bool,
    info: dict[str, Any],
    actions: list[np.ndarray],
    elapsed: float,
) -> None:
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.size:
        mean_action = action_array.reshape(len(actions), -1).mean(axis=0)
        min_action = action_array.reshape(len(actions), -1).min(axis=0)
        max_action = action_array.reshape(len(actions), -1).max(axis=0)
    else:
        mean_action = min_action = max_action = np.array([0.0, 0.0])
    print(
        f"[episode {episode_idx}] seed={seed} steps={step_count} "
        f"reward={episode_reward:.3f} terminal={terminal} truncated={truncated} "
        f"success={_info_get(info, 'success')} "
        f"target_dist={_fmt(_info_get(info, 'target_distance_3d'))} "
        f"target_reduction={_fmt(_info_get(info, 'target_distance_reduction'))} "
        f"path_left={_fmt(_info_get(info, 'path_length_to_target'))} "
        f"path_reduction={_fmt(_info_get(info, 'path_length_reduction'))} "
        f"path_ratio={_fmt(_info_get(info, 'path_ratio'))} "
        f"avg_speed={_fmt(_info_get(info, 'average translation speed'))} "
        f"trajectory={_fmt(_info_get(info, 'trajectory length'))} "
        f"fps={step_count / elapsed:.2f}"
    )
    print(
        "  action mean/min/max="
        f"{mean_action.round(3).tolist()} / "
        f"{min_action.round(3).tolist()} / "
        f"{max_action.round(3).tolist()}"
    )


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _info_get(info: dict[str, Any], name: str) -> Any:
    if name in info:
        return info[name]
    normalized = name.replace(" ", "_").lower()
    for key, value in info.items():
        if key.replace(" ", "_").lower() == normalized:
            return value
    return None


def _maybe_add_frame(
    frames: list[Any],
    args: argparse.Namespace,
    obs: dict[str, Any],
    step_idx: int,
    reward: float,
    terminal: bool,
    truncated: bool,
    action: np.ndarray | None,
) -> None:
    if args.gif is None or args.gif_every <= 0:
        return
    if step_idx % args.gif_every != 0 and not terminal and not truncated:
        return
    if len(frames) >= args.gif_max_frames:
        return
    frames.append(_draw_frame(obs, step_idx, reward, terminal, truncated, action))


def _maybe_add_mesh_frame(
    frames: list[Any],
    args: argparse.Namespace,
    image: np.ndarray | None,
    step_idx: int,
    *,
    force: bool = False,
) -> None:
    if args.mesh_gif is None or image is None:
        return
    if args.mesh_gif_every <= 0:
        return
    if not force and step_idx % args.mesh_gif_every != 0:
        return
    if len(frames) >= args.mesh_gif_max_frames:
        return
    frames.append(_mesh_frame_from_array(image))


def _mesh_frame_from_array(image: np.ndarray):
    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    return Image.fromarray(array[:, :, :3], mode="RGB")


def _draw_frame(
    obs: dict[str, Any],
    step_idx: int,
    reward: float,
    terminal: bool,
    truncated: bool,
    action: np.ndarray | None,
):
    from PIL import Image, ImageDraw

    width, height, margin = 640, 640, 56
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_box = (margin, margin, width - margin, height - margin)
    draw.rectangle(plot_box, outline=(210, 210, 210), width=1)
    draw.line((margin, height // 2, width - margin, height // 2), fill=(235, 235, 235))
    draw.line((width // 2, margin, width // 2, height - margin), fill=(235, 235, 235))

    tracking = _current_tracking(obs)
    target = np.asarray(obs["target"], dtype=np.float32).reshape(-1)[:2]
    points = [_to_pixel(point, width, height, margin) for point in tracking]
    if len(points) > 1:
        draw.line(points, fill=(28, 92, 184), width=4)
    for point in points:
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(28, 92, 184))

    tx, ty = _to_pixel(target, width, height, margin)
    draw.line((tx - 10, ty, tx + 10, ty), fill=(210, 45, 45), width=3)
    draw.line((tx, ty - 10, tx, ty + 10), fill=(210, 45, 45), width=3)
    draw.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), fill=(210, 45, 45))

    action_text = "action=None"
    if action is not None:
        flat_action = np.asarray(action).reshape(-1)
        action_text = f"action=[{flat_action[0]:.2f}, {flat_action[1]:.2f}]"
    status = f"step={step_idx} reward={float(reward):.3f} term={terminal} trunc={truncated}"
    draw.text((20, 16), status, fill=(20, 20, 20))
    draw.text((20, 36), action_text, fill=(20, 20, 20))
    draw.text((20, height - 30), "blue=tracking, red=target, coordinates normalized", fill=(80, 80, 80))
    return image


def _to_pixel(point: np.ndarray, width: int, height: int, margin: int) -> tuple[int, int]:
    x = float(np.clip(point[0], -1.0, 1.0))
    y = float(np.clip(point[1], -1.0, 1.0))
    px = margin + (x + 1.0) * 0.5 * (width - 2 * margin)
    py = margin + (1.0 - (y + 1.0) * 0.5) * (height - 2 * margin)
    return int(px), int(py)


def _save_gif(frames: list[Any], path: Path, fps: float) -> None:
    if not frames:
        print("No GIF frames were captured.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(1000 / max(fps, 0.1)), 1)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved GIF: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
