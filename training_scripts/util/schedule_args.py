def add_schedule_args(
    parser,
    *,
    heatup_steps,
    training_steps,
    eval_interval,
    explore_episodes,
    update_per_explore_step,
    eval_seed_count=None,
) -> None:
    parser.add_argument(
        "--heatup_steps",
        type=float,
        default=heatup_steps,
        help="Number of random heatup steps before training/evaluation.",
    )
    parser.add_argument(
        "--training_steps",
        type=float,
        default=training_steps,
        help="Total exploration steps to train for.",
    )
    parser.add_argument(
        "--eval_interval",
        type=float,
        default=eval_interval,
        help="Exploration steps between evaluations.",
    )
    parser.add_argument(
        "--explore_episodes",
        type=int,
        default=explore_episodes,
        help="Episodes collected between update phases.",
    )
    parser.add_argument(
        "--update_per_explore_step",
        type=float,
        default=update_per_explore_step,
        help="Gradient update steps per exploration step.",
    )
    parser.add_argument(
        "--eval_seed_count",
        type=int,
        default=eval_seed_count,
        help="Number of evaluation seeds to use. Use a small value for smoke tests.",
    )


def schedule_from_args(args):
    return {
        "heatup_steps": int(args.heatup_steps),
        "training_steps": int(args.training_steps),
        "eval_interval": int(args.eval_interval),
        "explore_episodes": int(args.explore_episodes),
        "update_per_explore_step": float(args.update_per_explore_step),
        "eval_seed_count": args.eval_seed_count,
    }
