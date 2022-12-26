from typing import Dict

MODELS_CONFIG: Dict[str, dict] = {
    "xxtiny": {
        "depths": [2, 2, 6, 2],
        "num_heads": [2, 4, 8, 16],
        "window_size": [7, 7, 14, 7],
        "dim": 64,
        "mlp_ratio": 3,
        "drop_path_rate": 0.2,
        "name": "xxtiny",
    },
    "xtiny": {
        "depths": [3, 4, 6, 5],
        "num_heads": [2, 4, 8, 16],
        "window_size": [7, 7, 14, 7],
        "dim": 64,
        "mlp_ratio": 3,
        "drop_path_rate": 0.2,
        "name": "xtiny",
    },
    "tiny": {
        "depths": [3, 4, 19, 5],
        "num_heads": [2, 4, 8, 16],
        "window_size": [7, 7, 14, 7],
        "dim": 64,
        "mlp_ratio": 3,
        "drop_path_rate": 0.2,
        "name": "tiny",
    },
    "small": {
        "depths": [3, 4, 19, 5],
        "num_heads": [3, 6, 12, 24],
        "window_size": [7, 7, 14, 7],
        "dim": 96,
        "mlp_ratio": 2,
        "drop_path_rate": 0.3,
        "layer_scale": 1e-5,
        "name": "small",
    },
    "base": {
        "depths": [3, 4, 19, 5],
        "num_heads": [4, 8, 16, 32],
        "window_size": [7, 7, 14, 7],
        "dim": 128,
        "mlp_ratio": 2,
        "drop_path_rate": 0.5,
        "layer_scale": 1e-5,
        "name": "base",
    },
}

TF_WEIGHTS_URL: str = (
    "https://github.com/EMalagoli92/GCViT-TensorFlow/releases/download"
)
