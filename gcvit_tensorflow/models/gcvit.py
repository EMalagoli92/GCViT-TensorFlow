from typing import List, Literal, Optional, Union

import tensorflow as tf
import tensorflow_addons as tfa

from gcvit_tensorflow import __version__
from gcvit_tensorflow.models.config import MODELS_CONFIG, TF_WEIGHTS_URL
from gcvit_tensorflow.models.layers.gcvit_layer import GCViTLayer
from gcvit_tensorflow.models.layers.patch_embed import PatchEmbed
from gcvit_tensorflow.models.layers.utils import (
    Identity_,
    LayerNorm_,
    Linear_,
    TruncNormalInitializer_,
)
from gcvit_tensorflow.models.utils import _to_channel_first


@tf.keras.utils.register_keras_serializable(package="gcvit")
class GCViT_(tf.keras.Model):
    """GCViT based on: "Hatamizadeh et al., Global Context Vision Transformers
    <https://arxiv.org/abs/2206.09959>"."""

    def __init__(
        self,
        dim: int,
        depths: List[int],
        window_size: List[int],
        mlp_ratio: float,
        num_heads: List[int],
        resolution: int = 224,
        drop_path_rate: float = 0.2,
        in_chans: int = 3,
        num_classes: int = 1000,
        qkv_bias: bool = True,
        qk_scale: Optional[bool] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        layer_scale: Optional[Union[int, float]] = None,
        classifier_activation: Optional[str] = None,
        data_format: Literal[
            "channels_first", "channels_last"
        ] = tf.keras.backend.image_data_format(),
        **kwargs,
    ):
        """
        Parameters
        ----------
        dim : int
            Feature size dimension.
        depths : List[int]
            Number of layers in each stage.
        window_size : List[int]
            Window size in each stage.
        mlp_ratio : float
            MLP ratio.
        num_heads : List[int]
            Number of heads in each stage.
        resolution : int, optional
            Input image resolution.
            The default is 224.
        drop_path_rate : float, optional
            Drop path rate.
            The default is 0.2.
        in_chans : int, optional
            Number of input channels.
            The default is 3.
        num_classes : int, optional
            Number of classes.
            The default is 1000.
        qkv_bias : bool, optional
            Bool argument for query, key, value learnable bias.
            The default is True.
        qk_scale : Optional[bool], optional
            Bool argument to scaling query, key.
            The default is None.
        drop_rate : float, optional
            Dropout rate.
            The default is 0.0.
        attn_drop_rate : float, optional
            Attention dropout rate.
            The default is 0.0.
        layer_scale : Optional[Union[int, float]], optional
            Layer scaling coefficient.
            The default is None.
        classifier_activation : Optional[str], optional
            String name for a tf.keras.layers.Activation layer.
            The default is None.
        data_format : Literal["channels_first", "channels_last"], optional
            A string, one of "channels_last" or "channels_first".
            The ordering of the dimensions in the inputs.
            "channels_last" corresponds to inputs with shape:
            (batch_size, height, width, channels)
            while "channels_first" corresponds to inputs with shape
            (batch_size, channels, height, width).
            The default is tf.keras.backend.image_data_format().
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.depths = depths
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.resolution = resolution
        self.drop_path_rate = drop_path_rate
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.layer_scale = layer_scale
        self.classifier_activation = classifier_activation
        self.data_format = data_format

        num_features = int(self.dim * 2 ** (len(self.depths) - 1))
        self.patch_embed = PatchEmbed(
            in_chans=self.in_chans, dim=self.dim, name="patch_embed"
        )
        self.pos_drop = tf.keras.layers.Dropout(rate=self.drop_rate, name="pos_drop")
        dpr = [
            i * self.drop_path_rate / (sum(self.depths) - 1)
            for i in range(sum(self.depths))
        ]
        self.levels = []
        for i in range(len(self.depths)):
            level = GCViTLayer(
                dim=int(self.dim * 2**i),
                depth=self.depths[i],
                num_heads=self.num_heads[i],
                window_size=self.window_size[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i]) : sum(self.depths[: i + 1])],
                downsample=(i < len(self.depths) - 1),
                layer_scale=self.layer_scale,
                input_resolution=int(2 ** (-2 - i) * self.resolution),
                name=f"levels/{i}",
            )
            self.levels.append(level)
        self.norm = LayerNorm_(num_features, name="norm")
        self.avgpool = tfa.layers.AdaptiveAveragePooling2D(
            output_size=1, data_format="channels_first", name="avgpool"
        )
        self.head = (
            Linear_(
                in_features=num_features,
                units=self.num_classes,
                kernel_initializer=TruncNormalInitializer_(std=0.02),
                bias_initializer=tf.keras.initializers.Zeros(),
                name="head",
            )
            if self.num_classes > 0
            else Identity_(name="head")
        )
        if self.classifier_activation is not None:
            self.classifier_activation_ = tf.keras.layers.Activation(
                self.classifier_activation, dtype=self.dtype, name="pred"
            )

    def call_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x)
        x = _to_channel_first(x)
        x = self.avgpool(x)
        x = tf.reshape(x, [-1, tf.math.reduce_prod(tf.shape(x[0]))])
        return x

    def call(self, inputs, *args, **kwargs):
        if self.data_format == "channels_last":
            inputs = _to_channel_first(inputs)
        x = self.call_features(inputs)
        x = self.head(x)
        if hasattr(self, "classifier_activation_"):
            x = self.classifier_activation_(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)

    def __to_functional(self):
        if self.built:
            x = tf.keras.layers.Input(shape=(self._build_input_shape[1:]))
            model = tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.name)
        else:
            raise ValueError(
                "This model has not yet been built. "
                "Build the model first by calling build() or "
                "by calling the model on a batch of data."
            )
        return model

    def summary(self, *args, **kwargs):
        self.__to_functional()
        super().summary(*args, **kwargs)

    def plot_model(self, *args, **kwargs):
        tf.keras.utils.plot_model(model=self.__to_functional(), *args, **kwargs)

    def save(self, *args, **kwargs):
        self.__to_functional().save(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "depths": self.depths,
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "num_heads": self.num_heads,
                "resolution": self.resolution,
                "drop_path_rate": self.drop_path_rate,
                "in_chans": self.in_chans,
                "num_classes": self.num_classes,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "layer_scale": self.layer_scale,
                "classifier_activation": self.classifier_activation,
                "data_format": self.data_format,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def GCViT(
    configuration: Optional[Literal["xxtiny", "xtiny", "tiny", "small", "base"]] = None,
    pretrained: bool = False,
    **kwargs,
) -> tf.keras.Model:
    """Wrapper function for GCViT model.

    Parameters
    ----------
    configuration : Optional[Literal["xxtiny", "xtiny", "tiny", "small", "base"]],
                    optional
        Name of GCViT predefined configuration.
        Possible values are: "xxtiny", "xtiny", "tiny", "small", "base".
        The default is None.
    pretrained : bool, optional
        Whether to use ImageNet pretrained weights.
        The default is False.
    **kwargs
        Additional keyword arguments.

    Raises
    ------
    KeyError
        If choosen configuration not in:
        ["xxtiny","xtiny","tiny","small","base"]

    Returns
    -------
    tf.keras.Model
        GCViT model.
    """
    if configuration is not None:
        if configuration in MODELS_CONFIG.keys():
            model = GCViT_(**MODELS_CONFIG[configuration], **kwargs)
            if pretrained:
                if model.data_format == "channels_last":
                    model.build((None, 224, 224, 3))
                elif model.data_format == "channels_first":
                    model.build((None, 3, 224, 224))
                weights_path = "{}/{}/{}.h5".format(
                    TF_WEIGHTS_URL, __version__, configuration
                )
                model_weights = tf.keras.utils.get_file(
                    fname="{}.h5".format(configuration),
                    origin=weights_path,
                    cache_subdir="datasets/gcvit_tensorflow",
                )
                model.load_weights(model_weights)
            return model
        else:
            raise KeyError(
                f"{configuration} configuration not found. "
                f"Valid values are: {list(MODELS_CONFIG.keys())}"
            )
    else:
        return GCViT_(**kwargs)
