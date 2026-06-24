from .nodes import Lotus2ModelLoader, Lotus2Inference

NODE_CLASS_MAPPINGS = {
    "Lotus2ModelLoader": Lotus2ModelLoader,
    "Lotus2Inference": Lotus2Inference
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lotus2ModelLoader": "Lotus-2 Model Loader",
    "Lotus2Inference": "Lotus-2 Predictor (Depth/Normal)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']