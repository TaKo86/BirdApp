"""train the Random Forest on the shared eight-species MFCC features."""

from sklearn.ensemble import RandomForestClassifier

from training_common import SETTINGS, run_sklearn, training_parser


def main() -> None:
    """run one RF experiment without changing the dataset split."""
    parser = training_parser("rf")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()
    if min(args.trees, args.jobs, args.min_samples_leaf) < 1 or (args.max_depth is not None and args.max_depth < 1):
        parser.error("Tree count, jobs, depth and minimum leaf size must be positive.")
    model = RandomForestClassifier(n_estimators=args.trees, n_jobs=args.jobs,
        max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced", random_state=SETTINGS["seed"])
    run_sklearn(args, model, "rf")


if __name__ == "__main__":
    main()
