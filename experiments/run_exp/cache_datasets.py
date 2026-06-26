"""
Cache HF datasets to CSV without running inference.
"""

from pathlib import Path
import sys
import argparse

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from grading_calibration.datasets import load_scientsbank, load_beetle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache HF datasets to CSV")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scientsbank", "beetle"],
        choices=["scientsbank", "beetle"],
        help="Datasets to cache (default: scientsbank beetle)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "scientsbank" in args.datasets:
        print("Caching SciEntsBank...")
        load_scientsbank(seed=args.seed)
    if "beetle" in args.datasets:
        print("Caching Beetle...")
        load_beetle(seed=args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
