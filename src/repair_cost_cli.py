from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from repair_cost_model import predict_costs_from_csv, train_model_from_csv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dataset() -> Path:
    return _repo_root() / "data" / "room_dataset.csv"


def _default_model_path() -> Path:
    return _repo_root() / "models" / "repair_cost_multimodal.joblib"


def _default_metrics_path() -> Path:
    return _repo_root() / "models" / "repair_cost_multimodal_metrics.json"


def _default_predictions_path() -> Path:
    return _repo_root() / "data" / "repair_verdicts.csv"


def _print_metrics(metrics: Dict[str, float]) -> None:
    print("Training finished.")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape_percent']:.2f}%")
    print(f"Rows: {int(metrics['rows'])}")
    print(f"Text branch weight: {metrics['text_weight']:.2f}")
    if int(metrics["target_is_synthetic"]) == 1:
        print("Target source: synthetic proxy cost (no explicit price column found).")
    else:
        print("Target source: explicit price column from dataset.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repair-cost",
        description=(
            "Multimodal repair cost estimator: uses image-derived CSV metrics plus text description when available."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    train_cmd = subparsers.add_parser("train", help="Train model and save to models/ folder.")
    train_cmd.add_argument("--dataset", type=Path, default=_default_dataset())
    train_cmd.add_argument("--model-path", type=Path, default=_default_model_path())
    train_cmd.add_argument("--metrics-path", type=Path, default=_default_metrics_path())
    train_cmd.add_argument("--target-column", type=str, default=None)
    train_cmd.add_argument("--text-column", type=str, default=None)
    train_cmd.add_argument("--test-size", type=float, default=0.2)
    train_cmd.add_argument("--random-state", type=int, default=42)

    predict_cmd = subparsers.add_parser("predict", help="Run inference for a CSV dataset.")
    predict_cmd.add_argument("--input-csv", type=Path, default=_default_dataset())
    predict_cmd.add_argument("--model-path", type=Path, default=_default_model_path())
    predict_cmd.add_argument("--output-csv", type=Path, default=_default_predictions_path())
    predict_cmd.add_argument("--text-column", type=str, default=None)

    return parser


def run_train(args: argparse.Namespace) -> int:
    metrics = train_model_from_csv(
        csv_path=args.dataset,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        target_column=args.target_column,
        text_column=args.text_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    _print_metrics(metrics)
    print(f"Model saved to: {args.model_path}")
    print(f"Metrics saved to: {args.metrics_path}")
    return 0


def run_predict(args: argparse.Namespace) -> int:
    result = predict_costs_from_csv(
        input_csv=args.input_csv,
        model_path=args.model_path,
        output_csv=args.output_csv,
        text_column=args.text_column,
    )
    print(f"Inference completed for {len(result)} rows.")
    print(f"Output saved to: {args.output_csv}")
    print(result[["predicted_repair_cost", "predicted_repair_cost_label"]].head(5).to_string(index=False))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        model_path = _default_model_path()
        if not model_path.exists():
            train_args = argparse.Namespace(
                dataset=_default_dataset(),
                model_path=model_path,
                metrics_path=_default_metrics_path(),
                target_column=None,
                text_column=None,
                test_size=0.2,
                random_state=42,
            )
            run_train(train_args)

        predict_args = argparse.Namespace(
            input_csv=_default_dataset(),
            model_path=model_path,
            output_csv=_default_predictions_path(),
            text_column=None,
        )
        return run_predict(predict_args)

    if args.command == "train":
        return run_train(args)
    if args.command == "predict":
        return run_predict(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
