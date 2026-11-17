"""train a scaled SVM on the same MFCC features used by the Random Forest."""

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from training_common import SETTINGS, run_sklearn, training_parser


def main() -> None:
    """default to a linear SVM so all recordings can be used practically."""
    parser = training_parser("svm")
    parser.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=10000)
    args = parser.parse_args()
    if args.c <= 0 or args.max_iter < 1:
        parser.error("C and max-iter must be positive.")
    # stop on non-convergence rather than silently report an unfinished fit
    warnings.filterwarnings("error", category=ConvergenceWarning)
    if args.kernel == "linear":
        svm = LinearSVC(C=args.c, dual=False, class_weight="balanced",
                        max_iter=args.max_iter, random_state=SETTINGS["seed"])
    else:
        print("RBF SVM may take a long time on the full dataset.", flush=True)
        svm = SVC(C=args.c, kernel="rbf", gamma="scale", class_weight="balanced",
                  cache_size=1024, max_iter=args.max_iter, random_state=SETTINGS["seed"])
    # the pipeline fits the scaler using training rows only
    run_sklearn(args, make_pipeline(StandardScaler(), svm), "svm")


if __name__ == "__main__":
    main()
