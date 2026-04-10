import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
APP_PC = ROOT / "app_pc"
sys.path.insert(0, str(APP_PC))

from src.aligned_inference import run_aligned_inference  # noqa: E402


def discover_models(models_dir: Path):
    stems = []
    for onnx_path in models_dir.glob("*.onnx"):
        stem = onnx_path.stem
        scaler = models_dir / f"{stem}_scaler.pkl"
        if scaler.exists():
            stems.append(stem)
    return sorted(set(stems))


def collect_dataset(data_dir: Path, limit_per_class: int | None = None):
    items = []
    for label_name, y in [("real", 0), ("fake", 1)]:
        class_dir = data_dir / label_name
        videos = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}])
        if limit_per_class is not None:
            videos = videos[:limit_per_class]
        items.extend((video, y) for video in videos)
    return items


def evaluate_model(model_stem: str, dataset: list[tuple[Path, int]], config_path: Path):
    y_true = []
    y_pred = []
    y_prob = []
    failures = []

    for video_path, label in dataset:
        try:
            result = run_aligned_inference(str(video_path), model_stem, str(config_path))
            y_true.append(label)
            y_pred.append(1 if result["prediction"] == "FAKE" else 0)
            y_prob.append(float(result["probability_fake"]))
        except Exception as exc:
            failures.append({"video": str(video_path), "error": str(exc)})

    metrics = {
        "samples": len(y_true),
        "failures": len(failures),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if y_true else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if y_true else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if y_true else 0.0,
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "failures_preview": failures[:10],
    }
    return metrics


def choose_best(results: dict):
    ranked = sorted(
        results.items(),
        key=lambda kv: (
            kv[1].get("f1", 0.0),
            kv[1].get("accuracy", 0.0),
            kv[1].get("auc", 0.0),
            kv[1].get("recall", 0.0),
        ),
        reverse=True,
    )
    return ranked[0][0] if ranked else None


def save_best_model(best_model: str, results: dict, out_path: Path):
    payload = {
        "best_model": best_model,
        "reason": "selected_by_offline_benchmark",
        "ranking_metric": ["f1", "accuracy", "auc", "recall"],
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sync_to_android(src: Path, android_models_dir: Path):
    android_models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, android_models_dir / src.name)


def main():
    parser = argparse.ArgumentParser(description="Benchmark offline ONNX models and choose the best default model.")
    parser.add_argument("--data-dir", default=str(APP_PC / "data"))
    parser.add_argument("--config", default=str(APP_PC / "config.yaml"))
    parser.add_argument("--models-dir", default=str(APP_PC / "models"))
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None, help="Optional model stems to evaluate")
    parser.add_argument("--sync-android", action="store_true", help="Copy best-model file to Android assets/models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_path = Path(args.config)
    models_dir = Path(args.models_dir)

    dataset = collect_dataset(data_dir, args.limit_per_class)
    if not dataset:
        raise SystemExit("No dataset files found in data/real and data/fake")

    model_stems = args.models if args.models else discover_models(models_dir)
    if not model_stems:
        raise SystemExit("No ONNX + scaler model pairs found")

    print(f"Dataset size: {len(dataset)} videos")
    print(f"Models: {', '.join(model_stems)}")

    results = {}
    for model_stem in model_stems:
        print(f"\n=== Benchmarking {model_stem} ===")
        metrics = evaluate_model(model_stem, dataset, config_path)
        results[model_stem] = metrics
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    best_model = choose_best(results)
    if not best_model:
        raise SystemExit("Could not determine best model")

    out_path = models_dir / "benchmark_best_model.json"
    save_best_model(best_model, results, out_path)
    print(f"\nBest model: {best_model}")
    print(f"Saved to: {out_path}")

    if args.sync_android:
        android_models_dir = ROOT / "app_dt" / "app" / "src" / "main" / "assets" / "models"
        sync_to_android(out_path, android_models_dir)
        print(f"Synced to Android assets: {android_models_dir / out_path.name}")


if __name__ == "__main__":
    main()
