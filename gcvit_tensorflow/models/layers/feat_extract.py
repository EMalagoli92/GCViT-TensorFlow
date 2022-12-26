import tensorflow as tf
import tensorflow_addons as tfa

from gcvit_tensorflow.models.layers.se import SE
from gcvit_tensorflow.models.layers.utils import Conv2d_, MaxPool2d_


@tf.keras.utils.register_keras_serializable(package="gcvit")
class FeatExtract(tf.keras.layers.Layer):
    """Feature extraction block based on: "Hatamizadeh et al., Global Context
    Vision Transformers <https://arxiv.org/abs/2206.09959>"."""

    def __init__(self, dim: int, keep_dim: bool = False, **kwargs):
        """
        Parameters
        ----------
        dim : int
            Feature size dimension.
        keep_dim : bool, optional
            Bool argument for maintaining the resolution.
            The default is False.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.keep_dim = keep_dim

    def build(self, input_shape):
        self.conv1 = Conv2d_(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.dim,
            bias=False,
            name="conv/0",
        )
        self.act1 = tfa.layers.GELU(approximate=False, name="conv/1")
        self.se = SE(inp=self.dim, oup=self.dim, name="conv/2")
        self.conv2 = Conv2d_(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            name="conv/3",
        )
        if not self.keep_dim:
            self.pool = MaxPool2d_(kernel_size=3, stride=2, padding=1, name="pool")
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        y = self.conv1(inputs)
        y = self.act1(y)
        y = self.se(y)
        y = self.conv2(y)
        x = inputs + y
        if not self.keep_dim:
            x = self.pool(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "keep_dim": self.keep_dim})
        return config
