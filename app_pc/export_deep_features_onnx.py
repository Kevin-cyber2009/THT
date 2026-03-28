import torch
import torchvision.models as models
import argparse
import os


def downgrade_onnx_ir(onnx_path: str, target_ir_version: int = 8):
    """
    Downgrade ONNX model IR version to be compatible with older runtimes.
    onnxruntime-android 1.16.3 supports max IR version 9.
    onnxruntime-android 1.19.0 supports IR version 10.
    Setting to 8 ensures maximum compatibility.
    """
    import onnx
    model = onnx.load(onnx_path)
    original_ir = model.ir_version
    model.ir_version = target_ir_version
    onnx.save(model, onnx_path)
    print(f"  IR version: {original_ir} → {target_ir_version}")


def convert_resnet50(output_path: str):
    """Convert ResNet50 sang ONNX với opset 11 và IR version 8"""
    print("Converting ResNet50...")

    model = models.resnet50(weights='IMAGENET1K_V1')
    # Remove final FC layer để lấy 2048-dim features
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
        opset_version=11,  # opset 11 → IR version 6-7, safe for all runtimes
        do_constant_folding=True,
        export_params=True,
    )

    # Ensure IR version compatibility
    downgrade_onnx_ir(output_path, target_ir_version=8)

    print(f"✓ ResNet50 saved to: {output_path}")
    _print_model_info(output_path)


def convert_efficientnet_b0(output_path: str):
    """Convert EfficientNet-B0 sang ONNX với opset 11 và IR version 8"""
    print("Converting EfficientNet-B0...")

    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # Replace classifier với Identity để lấy 1280-dim features
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
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
    )

    # Ensure IR version compatibility
    downgrade_onnx_ir(output_path, target_ir_version=8)

    print(f"✓ EfficientNet-B0 saved to: {output_path}")
    _print_model_info(output_path)


def _print_model_info(onnx_path: str):
    """Print model IR version and opset info"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        opset = model.opset_import[0].version if model.opset_import else "?"
        print(f"  IR version: {model.ir_version}, Opset: {opset}")
    except Exception as e:
        print(f"  (Could not read model info: {e})")


def verify_onnx_model(onnx_path: str):
    """Kiểm tra ONNX model có hoạt động không"""
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})

        out_flat = outputs[0].flatten()
        print(f"✓ Verification passed: {onnx_path}")
        print(f"  Input shape: (1, 3, 224, 224)")
        print(f"  Output shape: {outputs[0].shape}, flattened dim: {len(out_flat)}")
        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models to ONNX (IR version 8, opset 11)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./onnx_models',
        help='Output directory for ONNX models'
    )
    parser.add_argument(
        '--ir_version',
        type=int,
        default=8,
        help='Target ONNX IR version (default: 8, safe for onnxruntime-android >= 1.16)'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    resnet_path = os.path.join(args.output_dir, 'resnet50_features.onnx')
    efficientnet_path = os.path.join(args.output_dir, 'efficientnet_b0_features.onnx')

    print("=" * 60)
    print("Converting models with opset=11, IR version=8")
    print("Compatible with onnxruntime-android >= 1.16.3")
    print("=" * 60)
    print()

    convert_resnet50(resnet_path)
    print()
    convert_efficientnet_b0(efficientnet_path)

    print()
    print("=" * 60)
    print("Verifying ONNX models (PC onnxruntime)...")
    print("=" * 60)
    ok1 = verify_onnx_model(resnet_path)
    ok2 = verify_onnx_model(efficientnet_path)

    print()
    print("=" * 60)
    if ok1 and ok2:
        print("✅ Both models OK!")
    else:
        print("⚠️  Some models failed verification")
    print()
    print("Next steps:")
    print(f"  1. Copy files from {args.output_dir} to:")
    print("     app_dt/app/src/main/assets/models/")
    print("  2. Rebuild the Android app")
    print("=" * 60)


if __name__ == '__main__':
    main()