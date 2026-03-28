import argparse
import numpy as np
import joblib
import sys

def check_onnx_model(onnx_path: str, scaler_path: str = None):
    """Kiểm tra ONNX model và in ra định dạng output."""
    
    try:
        import onnxruntime as rt
    except ImportError:
        print("❌ Cần cài đặt onnxruntime: pip install onnxruntime")
        sys.exit(1)
    
    print("=" * 70)
    print("ONNX MODEL OUTPUT CHECKER")
    print("=" * 70)
    
    print(f"\n📦 Loading ONNX model: {onnx_path}")
    try:
        sess = rt.InferenceSession(onnx_path)
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        sys.exit(1)
    
    print(f"✅ Model loaded thành công")
    print(f"   Input names: {sess.get_inputs()}")
    print(f"   Output names: {sess.get_outputs()}")
    
    n_features = None
    if scaler_path:
        print(f"\n📦 Loading scaler: {scaler_path}")
        try:
            scaler_data = joblib.load(scaler_path)
            scaler = scaler_data.get('scaler')
            if scaler and hasattr(scaler, 'n_features_in_'):
                n_features = scaler.n_features_in_
                print(f"✅ Scaler có {n_features} features")
        except Exception as e:
            print(f"⚠️ Không thể load scaler: {e}")
    
    if n_features is None:
        for inp in sess.get_inputs():
            shape = inp.shape
            if len(shape) >= 2 and shape[1] is not None:
                n_features = shape[1]
                print(f"✅ Đoán n_features từ input shape: {n_features}")
                break
    
    if n_features is None:
        n_features = 39 
        print(f"⚠️ Không xác định được n_features, dùng default: {n_features}")
    
    print(f"\n🧪 Testing với dummy input (1, {n_features})...")
    dummy = np.zeros((1, n_features), dtype=np.float32)
    
    if scaler_path and 'scaler' in locals() and scaler is not None:
        dummy = scaler.transform(dummy).astype(np.float32)
        print("✅ Đã apply scaler transform")
    
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: dummy})
    
    print(f"\n📊 OUTPUT ANALYSIS:")
    print(f"   Số lượng outputs: {len(outputs)}")
    
    for i, out in enumerate(outputs):
        print(f"\n   Output[{i}]:")
        print(f"      Type: {type(out)}")
        
        if isinstance(out, np.ndarray):
            print(f"      Shape: {out.shape}")
            print(f"      Dtype: {out.dtype}")
            print(f"      Values: {out}")
            
        elif isinstance(out, list):
            print(f"      Length: {len(out)}")
            if len(out) > 0:
                print(f"      First element type: {type(out[0])}")
                print(f"      First element: {out[0]}")
                
                # Kiểm tra nếu là List<Map>
                if isinstance(out[0], dict):
                    print(f"      ✅ Đây là List<Map<*, *>> format (LightGBM ONNX standard)")
                    print(f"      Keys: {list(out[0].keys())}")
                    print(f"      Values: {list(out[0].values())}")
                    
        elif isinstance(out, dict):
            print(f"      Keys: {list(out.keys())}")
            print(f"      Values: {list(out.values())}")
            print(f"      ✅ Đây là Map<*, *> format")
            
        else:
            print(f"      Value: {out}")
    
    print(f"\n🔍 LABEL & PROBABILITY ANALYSIS:")
    
    if len(outputs) >= 2:
        label_out = outputs[0]
        prob_out = outputs[1]
        
        label = None
        if isinstance(label_out, np.ndarray):
            label = int(label_out.flatten()[0])
        elif isinstance(label_out, list) and len(label_out) > 0:
            label = int(label_out[0])
        
        print(f"   Label: {label}")
        
        prob_fake = None
        
        if isinstance(prob_out, list) and len(prob_out) > 0:
            if isinstance(prob_out[0], dict):
                prob_map = prob_out[0]
                # Thử các key khác nhau
                for key in [1, 1, "1", 1.0]:
                    if key in prob_map:
                        prob_fake = float(prob_map[key])
                        break
                if prob_fake is None:
                    values = [float(v) for v in prob_map.values() if isinstance(v, (int, float, np.number))]
                    if len(values) >= 2:
                        prob_fake = values[1]
                    elif len(values) == 1:
                        prob_fake = values[0]
                        
        elif isinstance(prob_out, np.ndarray):
            flat = prob_out.flatten()
            if len(flat) >= 2:
                prob_fake = float(flat[1])
            elif len(flat) == 1:
                prob_fake = float(flat[0])
        
        if prob_fake is not None:
            print(f"   Probability FAKE: {prob_fake:.4f}")
            print(f"   Probability REAL: {1.0 - prob_fake:.4f}")
        else:
            print(f"   ⚠️ Không thể parse probability")
            if label is not None:
                print(f"   Fallback: label={label} -> prob_fake={0.9 if label == 1 else 0.1}")
    
    print("\n" + "=" * 70)
    print("✅ CHECK COMPLETE")
    print("=" * 70)
    
    print("\n💡 KHUYẾN NGHỊ:")
    print("   - Nếu output là List<Map<*, *>>, Android app đã hỗ trợ")
    print("   - Nếu output là FloatArray, Android app đã hỗ trợ")
    print("   - Nếu gặp lỗi khác, hãy kiểm tra log của Android app")
    
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra ONNX model output format")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--scaler", help="Path to scaler.pkl file (optional)")
    args = parser.parse_args()
    
    check_onnx_model(args.model, args.scaler)