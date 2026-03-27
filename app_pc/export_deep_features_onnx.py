import argparse
import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


def export_resnet50_onnx(output_path: str):
    """Export ResNet50 (feature extraction - remove final FC layer) to ONNX"""
    print("=" * 70)
    print("Exporting ResNet50 for feature extraction -> ONNX")
    print("=" * 70)
    
    model = models.resnet50(weights='IMAGENET1K_V1')

    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ResNet50 ONNX saved: {output_path}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX model verified")
    print(f"   Input shape: (batch, 3, 224, 224)")
    print(f"   Output shape: (batch, 2048, 1, 1) -> flatten to (batch, 2048)")
    
    return output_path


def export_efficientnet_b0_onnx(output_path: str):
    """Export EfficientNet-B0 (feature extraction) to ONNX"""
    print("=" * 70)
    print("Exporting EfficientNet-B0 for feature extraction -> ONNX")
    print("=" * 70)
    
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    model.classifier = nn.Identity()
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    
    print(f"✅ EfficientNet-B0 ONNX saved: {output_path}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX model verified")
    print(f"   Input shape: (batch, 3, 224, 224)")
    print(f"   Output shape: (batch, 1280)")
    
    return output_path


def test_onnx_inference(onnx_path: str, model_type: str = 'resnet50'):
    """Test ONNX inference"""
    print(f"\n🧪 Testing ONNX inference: {model_type}")
    
    import onnxruntime as rt
    
    sess = rt.InferenceSession(onnx_path)
    
    # Test with dummy input
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Normalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    dummy = (dummy - mean) / std
    
    output = sess.run(None, {'input': dummy})
    features = output[0]
    
    print(f"✅ ONNX inference successful")
    print(f"   Input shape: {dummy.shape}")
    print(f"   Output shape: {features.shape}")
    print(f"   Feature stats: mean={features.mean():.4f}, std={features.std():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export deep feature models to ONNX")
    parser.add_argument("--out_dir", default="app_pc/models", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("EXPORT DEEP FEATURE MODELS TO ONNX")
    print("=" * 70 + "\n")
    
    # Export ResNet50
    resnet_path = os.path.join(args.out_dir, "resnet50_features.onnx")
    export_resnet50_onnx(resnet_path)
    test_onnx_inference(resnet_path, 'resnet50')
    
    print("\n")
    
    efficientnet_path = os.path.join(args.out_dir, "efficientnet_b0_features.onnx")
    export_efficientnet_b0_onnx(efficientnet_path)
    test_onnx_inference(efficientnet_path, 'efficientnet_b0')
    
    print("\n" + "=" * 70)
    print("✅ ALL DONE! Copy these files to Android assets/models/:")
    print(f"   {resnet_path}")
    print(f"   {efficientnet_path}")
    print("=" * 70)