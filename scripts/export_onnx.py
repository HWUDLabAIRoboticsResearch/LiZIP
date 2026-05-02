import torch
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.python.model import PointPredictorMLP

def export_to_onnx(model_path, onnx_path, context_size=3, hidden_dim=256):
    print(f"Loading model from {model_path}...")
    model = PointPredictorMLP(context_size=context_size, hidden_dim=hidden_dim)
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    dummy_input = torch.randn(1, context_size * 3)
    
    print(f"Exporting to ONNX: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    os.makedirs("models/onnx", exist_ok=True)
    
    # Default model
    export_to_onnx(
        "models/grid_search/mlp_c3_h256.pth", 
        "models/onnx/mlp_c3_h256.onnx",
        context_size=3,
        hidden_dim=256
    )
