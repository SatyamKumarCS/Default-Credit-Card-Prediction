import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.predict import load_model  # noqa: E402

def test_model_loading():
    """Test to ensure that the trained model exists and can be loaded properly."""
    model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    assert os.path.exists(model_path), "Model file is missing!"
    
    # Just loading it to ensure it's unpickled properly
    model = load_model(model_path)
    assert model is not None, "Model failed to load"
