import joblib
import numpy as np

MODEL_IN  = r"D:\AI\app_pc\models\straw.pkl"
MODEL_OUT = r"D:\AI\app_pc\models\straw.onnx"

print("📦 Loading straw.pkl...")
data = joblib.load(MODEL_IN)

model         = data['model']        
scaler        = data['scaler']
calibrator    = data.get('calibrator')
feature_names = data.get('feature_names')

print(f"✅ Model type: {type(model).__name__}")
print(f"✅ Feature names: {feature_names}")

n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else len(feature_names or [])
print(f"✅ Số features: {n_features}")

from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

initial_types = [("float_input", FloatTensorType([None, n_features]))]

print("🔄 Converting LightGBM → ONNX...")
onnx_model = convert_lightgbm(
    model,
    initial_types=initial_types,
    target_opset=12
)

import onnx
onnx.save(onnx_model, MODEL_OUT)
print(f"✅ Đã lưu: {MODEL_OUT}")

import joblib, pickle
scaler_data = {
    'scaler':        scaler,
    'calibrator':    calibrator,   
    'feature_names': feature_names,
}
joblib.dump(scaler_data, "models/alpha_scaler.pkl")
print("✅ Đã lưu: models/alpha_scaler.pkl")

print("\n🧪 Test ONNX model...")
import onnxruntime as rt  

sess   = rt.InferenceSession(MODEL_OUT)
dummy  = np.zeros((1, n_features), dtype=np.float32)
dummy  = scaler.transform(dummy).astype(np.float32)
output = sess.run(None, {"float_input": dummy})
print(f"✅ ONNX output: label={output[0]}, proba={output[1]}")
print("\n🎉 Convert thành công! Copy 2 file vào assets/models/:")
print(f"   - models/straw.onnx")
print(f"   - models/straw.pkl")