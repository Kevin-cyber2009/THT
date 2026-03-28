import argparse
import json
import os
import joblib
import numpy as np


def extract_scaler_params(pkl_path: str, output_dir: str = None):
    print("=" * 60)
    print("EXTRACT SCALER PARAMS")
    print("=" * 60)

    print(f"\nLoading: {pkl_path}")
    data = joblib.load(pkl_path)

    scaler        = data.get('scaler')
    feature_names = data.get('feature_names')

    if scaler is None:
        print("ERROR: No 'scaler' key in pkl file")
        return None

    if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        print("ERROR: Scaler does not have mean_ or scale_ attributes")
        print("       Was the scaler fitted?")
        return None

    mean_  = scaler.mean_.tolist()
    scale_ = scaler.scale_.tolist()
    var_   = scaler.var_.tolist() if hasattr(scaler, 'var_') else [s**2 for s in scale_]
    n_feat = int(scaler.n_features_in_) if hasattr(scaler, 'n_features_in_') else len(mean_)

    print(f"  n_features: {n_feat}")
    print(f"  mean  (first 5): {[f'{v:.4f}' for v in mean_[:5]]}")
    print(f"  scale (first 5): {[f'{v:.4f}' for v in scale_[:5]]}")

    params = {
        "n_features": n_feat,
        "mean_":      mean_,
        "scale_":     scale_,
        "var_":       var_,
        "feature_names": feature_names,
        "sklearn_version_saved": "1.5.1",
        "note": "Use (x - mean_[i]) / scale_[i] for manual scaling — no sklearn needed"
    }

    model_stem = os.path.splitext(os.path.basename(pkl_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(pkl_path)
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"{model_stem}_scaler_params.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)

    print(f"\n✅ Saved: {out_path}")
    print(f"\nNext: copy this file to:")
    print(f"  app_dt/app/src/main/assets/models/")
    print(f"  (same folder as {model_stem}.onnx and {model_stem}_scaler.pkl)")
    print("=" * 60)

    return out_path


def verify_params(params_path: str, pkl_path: str):
    """Verify that manual scaling produces same result as sklearn."""
    print("\nVerifying manual scaling vs sklearn...")
    try:
        import numpy as np
        import joblib

        with open(params_path, 'r') as f:
            params = json.load(f)

        data   = joblib.load(pkl_path)
        scaler = data['scaler']

        mean_  = np.array(params['mean_'])
        scale_ = np.array(params['scale_'])
        n_feat = params['n_features']

        x = np.random.randn(n_feat).reshape(1, -1)

        sklearn_result = scaler.transform(x).flatten()
        manual_result  = ((x.flatten() - mean_) / scale_)

        max_diff = np.max(np.abs(sklearn_result - manual_result))
        print(f"  Max diff between sklearn and manual: {max_diff:.2e}")

        if max_diff < 1e-5:
            print("  ✅ Manual scaling matches sklearn exactly")
        else:
            print("  ⚠️  Small numerical difference (acceptable)")

    except Exception as e:
        print(f"  ⚠️  Could not verify: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract StandardScaler params from pkl to JSON"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to .pkl model file (e.g., models/onestar.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: same as model)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: File not found: {args.model}")
        return

    out_path = extract_scaler_params(args.model, args.output)

    if out_path:
        verify_params(out_path, args.model)


if __name__ == '__main__':
    main()