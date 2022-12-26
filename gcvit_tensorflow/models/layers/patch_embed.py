import tensorflow as tf

from gcvit_tensorflow.models.layers.reduce_size import ReduceSize
from gcvit_tensorflow.models.layers.utils import Conv2d_
from gcvit_tensorflow.models.utils import _to_channel_last


@tf.keras.utils.register_keras_serializable(package="gcvit")
class PatchEmbed(tf.keras.layers.Layer):
    """Patch embedding block based on: "Hatamizadeh et al., Global Context
    Vision Transformers <https://arxiv.org/abs/2206.09959>"."""

    def __init__(self, in_chans: int = 3, dim: int = 96, **kwargs):
        """
        Parameters
        ----------
        in_chans : int, optional
            Number of input channels.
            The default is 3.
        dim : int, optional
            Feature size dimension.
            The default is 96.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_chans = in_chans
        self.dim = dim

    def build(self, input_shape):
        self.proj = Conv2d_(
            in_channels=self.in_chans,
            out_channels=self.dim,
            kernel_size=3,
            stride=2,
            padding=1,
            name="proj",
        )
        self.conv_down = ReduceSize(dim=self.dim, keep_dim=True, name="conv_down")
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.proj(inputs)
        x = _to_channel_last(x)
        x = self.conv_down(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"in_chans": self.in_chans, "dim": self.dim})
        return config
