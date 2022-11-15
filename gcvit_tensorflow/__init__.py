import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
import tensorflow.experimental.numpy as tnp

tnp.experimental_enable_numpy_behavior()
import json
import random

import numpy as np
from tensorflow.python.framework import random_seed

# Set Seed
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
random_seed.set_seed(SEED)
np.random.seed(SEED)

with open(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "version.json")), "r"
) as handle:
    __version__ = json.load(handle)["version"]

from .models.gcvit import GCViT
