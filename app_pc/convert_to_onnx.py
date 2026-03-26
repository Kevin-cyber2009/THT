import argparse
import os
import joblib
import numpy as np

DEFAULT_PKL_IN   = r"app_pc/models/x.pkl"  
DEFAULT_ONNX_OUT = r"app_pc/models/x.onnx"  

def convert(pkl_path: str, onnx_path: str):
    print("=" * 70)
    print("CONVERT PKL → ONNX + SCALER")
    print("=" * 70)

    print(f"\n📦 Loading {pkl_path} ...")
    data = joblib.load(pkl_path)

    model         = data['model']
    scaler        = data['scaler']
    calibrator    = data.get('calibrator')
    feature_names = data.get('feature_names')

    print(f"✅ Model type   : {type(model).__name__}")
    print(f"✅ Feature names: {feature_names}")

    n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else (
        len(feature_names) if feature_names else None
    )
    print(f"✅ n_features    : {n_features}")

    if feature_names is None:
        print("⚠️  feature_names không có trong pkl — dùng fallback order")
        traditional = [
            'fft_mean', 'fft_std', 'fft_max', 'fft_high_freq_energy', 'fft_radial_slope',
            'dct_mean', 'dct_std', 'dct_dc_mean', 'dct_ac_energy',
            'prnu_mean', 'prnu_std', 'prnu_autocorr', 'prnu_temporal_consistency',
            'flow_mean_magnitude', 'flow_std_magnitude', 'flow_smoothness', 'flow_temporal_consistency',
            'entropy_mean', 'entropy_std', 'entropy_slope',
            'fractal_dim_mean', 'fractal_dim_std',
            'causal_prediction_error', 'causal_predictability',
            'compression_mean', 'compression_std', 'compression_delta_mean', 'complexity_mean',
        ]
        deep = [
            'deep_feat_mean', 'deep_feat_std', 'deep_feat_max', 'deep_feat_min',
            'deep_temporal_var_mean', 'deep_temporal_var_std',
            'deep_l2_norm_mean', 'deep_l2_norm_std',
            'deep_similarity_mean', 'deep_similarity_std',
            'deep_sparsity',
        ]
        if n_features and n_features > 28:
            feature_names = traditional + deep
        else:
            feature_names = traditional
        print(f"   → Dùng fallback {len(feature_names)} features")

    try:
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        print("❌ Cần cài: pip install onnxmltools lightgbm skl2onnx")
        raise

    initial_types = [("float_input", FloatTensorType([None, n_features]))]

    print(f"\n🔄 Converting LightGBM ({n_features} features) → ONNX ...")
    onnx_model = convert_lightgbm(
        model,
        initial_types=initial_types,
        target_opset=12
    )

    import onnx
    onnx.save(onnx_model, onnx_path)
    print(f"✅ ONNX saved   : {onnx_path}")

    model_stem   = os.path.splitext(os.path.basename(onnx_path))[0]
    scaler_path  = os.path.join(os.path.dirname(onnx_path), f"{model_stem}_scaler.pkl")

    scaler_data = {
        'scaler':        scaler,
        'calibrator':    calibrator,
        'feature_names': feature_names,   # FIX: lưu đúng thứ tự
        'n_features':    n_features,
    }
    joblib.dump(scaler_data, scaler_path)
    print(f"✅ Scaler saved : {scaler_path}")
    print(f"   feature_names ({len(feature_names)}): {feature_names[:5]} ... {feature_names[-3:]}")

    # ── Test ONNX ─────────────────────────────────────────────────────────────
    print(f"\n🧪 Testing ONNX inference ...")
    import onnxruntime as rt

    sess   = rt.InferenceSession(onnx_path)
    dummy  = np.zeros((1, n_features), dtype=np.float32)
    dummy  = scaler.transform(dummy).astype(np.float32)
    output = sess.run(None, {"float_input": dummy})
    print(f"✅ ONNX output  : label={output[0]}, proba keys={list(output[1][0].keys()) if output[1] else '?'}")

    # ── Test consistency: PC predict vs ONNX predict ─────────────────────────
    print(f"\n🔁 Consistency check (PC scaler → ONNX) ...")
    test_input = np.random.rand(5, n_features).astype(np.float32)
    test_scaled = scaler.transform(test_input).astype(np.float32)

    onnx_labels = []
    for i in range(5):
        out = sess.run(None, {"float_input": test_scaled[i:i+1]})
        onnx_labels.append(int(out[0][0]))

    # Calibrated prediction từ calibrator (nếu có)
    if calibrator is not None:
        from sklearn.calibration import CalibratedClassifierCV
        cal_probs = calibrator.predict_proba(test_scaled)[:, 1]
        cal_preds = (cal_probs >= 0.5).astype(int).tolist()
        matches = sum(a == b for a, b in zip(onnx_labels, cal_preds))
        print(f"   ONNX labels      : {onnx_labels}")
        print(f"   Calibrated labels: {cal_preds}")
        print(f"   Match: {matches}/5")
        if matches < 4:
            print("   ⚠️  Nhiều mismatch — ONNX dùng base model, không qua calibrator")
            print("       Điều này bình thường. ONNX predict raw LightGBM.")
    else:
        print(f"   ONNX labels: {onnx_labels} (không có calibrator để so sánh)")

    print("\n" + "=" * 70)
    print("✅ DONE! Copy 2 file sau vào assets/models/ của Android:")
    print(f"   {onnx_path}")
    print(f"   {scaler_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_PKL_IN,  help="Input .pkl path")
    parser.add_argument("--out",   default=DEFAULT_ONNX_OUT, help="Output .onnx path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    convert(args.model, args.out)