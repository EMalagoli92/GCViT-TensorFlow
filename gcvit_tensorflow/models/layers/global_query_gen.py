import tensorflow as tf

from gcvit_tensorflow.models.layers.feat_extract import FeatExtract
from gcvit_tensorflow.models.utils import _to_channel_last


@tf.keras.utils.register_keras_serializable(package="gcvit")
class GlobalQueryGen(tf.keras.layers.Layer):
    """Global query generator based on: "Hatamizadeh et al., Global Context
    Vision Transformers <https://arxiv.org/abs/2206.09959>"."""

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        window_size: int,
        num_heads: int,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dim : int
            Feature size dimension.
        input_resolution : int
            Input image resolution.
        window_size : int
            Window size.
        num_heads : int
            Number of heads.
        **kwargs
            Additional keyword arguments.

        For instance, repeating log(56/7) = 3 blocks, with input window
        dimension 56 and output window dimension 7 at down-sampling ratio 2.
        Please check Fig.5 of GC ViT paper for details.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads

    def build(self, input_shape):
        if self.input_resolution == 56:
            self.to_q_global = [
                FeatExtract(self.dim, keep_dim=False, name=f"to_q_global/{i}")
                for i in range(3)
            ]
        elif self.input_resolution == 28:
            self.to_q_global = [
                FeatExtract(self.dim, keep_dim=False, name=f"to_q_global/{i}")
                for i in range(2)
            ]
        elif self.input_resolution == 14:
            if self.window_size == 14:
                self.to_q_global = [
                    FeatExtract(self.dim, keep_dim=True, name=f"to_q_global/{i}")
                    for i in range(1)
                ]
            elif self.window_size == 7:
                self.to_q_global = [
                    FeatExtract(self.dim, keep_dim=False, name=f"to_q_global/{i}")
                    for i in range(1)
                ]
        elif self.input_resolution == 7:
            self.to_q_global = [
                FeatExtract(self.dim, keep_dim=True, name=f"to_q_global/{i}")
                for i in range(1)
            ]

        self.N = self.window_size * self.window_size
        self.dim_head = self.dim // self.num_heads
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.to_q_global:
            x = layer(x)
        x = _to_channel_last(x)
        B = tf.shape(x)[0]
        x = tf.reshape(x, (B, 1, self.N, self.num_heads, self.dim_head))
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4])
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "input_resolution": self.input_resolution,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
            }
        )
        return config
