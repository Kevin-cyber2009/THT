"""
Script để convert PyTorch deep learning models sang ONNX cho mobile app.

Cách dùng:
    python convert_deep_models_to_onnx.py --output_dir ./onnx_models

Sau đó copy các file .onnx vào app_dt/app/src/main/assets/models/
"""

import torch
import torchvision.models as models
import argparse
import os


def convert_resnet50(output_path: str):
    """Convert ResNet50 sang ONNX"""
    print("Converting ResNet50...")
    
    model = models.resnet50(weights='IMAGENET1K_V1')
    # Remove final FC layer để lấy features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11
    )
    
    print(f"✓ ResNet50 saved to: {output_path}")


def convert_efficientnet_b0(output_path: str):
    """Convert EfficientNet-B0 sang ONNX"""
    print("Converting EfficientNet-B0...")
    
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # Replace classifier với Identity để lấy features
    model.classifier = torch.nn.Identity()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11
    )
    
    print(f"✓ EfficientNet-B0 saved to: {output_path}")


def verify_onnx_model(onnx_path: str):
    """Kiểm tra ONNX model có hoạt động không"""
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        
        # Test inference
        import numpy as np
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"✓ Verification passed: {onnx_path}")
        print(f"  Input shape: (1, 3, 224, 224)")
        print(f"  Output shape: {outputs[0].shape}")
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                        help='Output directory for ONNX models')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert models
    resnet_path = os.path.join(args.output_dir, 'resnet50_features.onnx')
    efficientnet_path = os.path.join(args.output_dir, 'efficientnet_b0_features.onnx')
    
    convert_resnet50(resnet_path)
    convert_efficientnet_b0(efficientnet_path)
    
    # Verify
    print("\nVerifying ONNX models...")
    verify_onnx_model(resnet_path)
    verify_onnx_model(efficientnet_path)
    
    print(f"\n✓ Done! Copy các file .onnx từ {args.output_dir} vào app_dt/app/src/main/assets/models/")


if __name__ == '__main__':
    main()