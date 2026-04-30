import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_WANDB_ENTITY = "stEVE_training"
DEFAULT_WANDB_PROJECT = "stEVE_trainin"
DEFAULT_WANDB_GROUP = "stEVE_training_pipeline"


def add_wandb_args(parser) -> None:
    load_env_file()
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases tracking for this run.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY),
        help="W&B entity/workspace.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=os.environ.get("WANDB_GROUP", DEFAULT_WANDB_GROUP),
        help="W&B group name used to organize related runs.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
        help="W&B run mode.",
    )


def init_wandb(
    args,
    *,
    run_name: str,
    config: Dict[str, Any],
    tags: Optional[Iterable[str]] = None,
):
    load_env_file()
    if args.no_wandb or args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases is not installed. Run `python3 -m pip install wandb` "
            "or pass `--no_wandb` to train without tracking."
        ) from exc

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=False)

    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        tags=list(tags or []),
        config=_sanitize_config(config),
        mode=args.wandb_mode,
    )


def finish_wandb(wandb_run) -> None:
    if wandb_run is not None:
        wandb_run.finish()


def load_env_file(env_file: str = ".env") -> None:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / env_file, repo_root / env_file]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        for line in candidate.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def _sanitize_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
