import os
from pathlib import Path
MODEL_PATH = Path(__file__).parent.joinpath("models","gnn.pth")
try:
    from gnn_impl import GNNWrapper
    GNN_IMPL_AVAILABLE = True
except Exception:
    GNN_IMPL_AVAILABLE = False
MODEL = None
def load_gnn(model_path=None):
    global MODEL
    if MODEL is not None:
        return MODEL
    if model_path is None:
        model_path = MODEL_PATH
    if not model_path.exists():
        return None
    if not GNN_IMPL_AVAILABLE:
        return None
    try:
        import torch
        MODEL = GNNWrapper()
        MODEL.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        MODEL.eval()
        return MODEL
    except Exception:
        return None
def predict_with_gnn(claims, evidence, params=None):
    m = load_gnn()
    if m is None:
        return None
    try:
        out = m.predict(claims, evidence, params or {})
        return out
    except Exception:
        return None
