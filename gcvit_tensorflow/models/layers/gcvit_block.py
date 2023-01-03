from typing import Literal, Optional, Union

import tensorflow as tf

from gcvit_tensorflow.models.layers.mlp import Mlp
from gcvit_tensorflow.models.layers.utils import DropPath_, Identity_, LayerNorm_
from gcvit_tensorflow.models.layers.window_attention import WindowAttention
from gcvit_tensorflow.models.layers.window_attention_global import WindowAttentionGlobal
from gcvit_tensorflow.models.utils import window_partition, window_reverse


@tf.keras.utils.register_keras_serializable(package="gcvit")
class GCViTBlock(tf.keras.layers.Layer):
    """GCViT block based on: "Hatamizadeh et al., Global Context Vision
    Transformers <https://arxiv.org/abs/2206.09959>"."""

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[bool] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "gelu",
        attention: Literal[
            "WindowAttention", "WindowAttentionGlobal"
        ] = "WindowAttentionGlobal",
        layer_scale: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dim : int
            Feature size dimension.
        input_resolution : int
            Input image resolution.
        num_heads : int
            Number of attention head.
        window_size : int, optional
            Window size.
            The default is 7.
        mlp_ratio : float, optional
            MLP ratio.
            The default is 4.0.
        qkv_bias : bool, optional
            Bool argument for query, key, value learnable bias.
            The default is True.
        qk_scale : Optional[bool], optional
            Bool argument to scaling query, key.
            The default is None.
        drop : float, optional
            Dropout rate.
            The default is 0.0.
        attn_drop : float, optional
            Attention dropout rate.
            The default is 0.0.
        drop_path : float, optional
            Drop path rate.
            The default is 0.0.
        act_layer : str, optional
            Name of activation layer.
            The default is "gelu".
        attention : Literal["WindowAttention", "WindowAttentionGlobal"], optional
            Attention block type.
            The default is "WindowAttentionGlobal".
        layer_scale : Optional[Union[int, float]], optional
            Layer scaling coefficient.
            The default is None.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.attention = attention
        self.layer_scale = layer_scale

    def build(self, input_shape):
        self.norm1 = LayerNorm_(self.dim, name="norm1")
        if self.attention == "WindowAttention":
            self.attn = WindowAttention
        elif self.attention == "WindowAttentionGlobal":
            self.attn = WindowAttentionGlobal
        else:
            raise ValueError(f"Unknown Attention block type: {self.attention}")
        self.attn = self.attn(
            self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            name="attn",
        )
        self._drop_path = (
            DropPath_(self.drop_path, name="drop_path")
            if self.drop_path > 0.0
            else Identity_(name="drop_path")
        )
        self.norm2 = LayerNorm_(self.dim, name="norm2")
        self.mlp = Mlp(
            in_features=self.dim,
            hidden_features=int(self.dim * self.mlp_ratio),
            act_layer=self.act_layer,
            drop=self.drop,
            name="mlp",
        )
        self._layer_scale = False
        if self.layer_scale is not None and type(self.layer_scale) in [int, float]:
            self._layer_scale = True
            self.gamma1 = self.add_weight(
                name="gamma1",
                shape=[self.dim],
                initializer=tf.keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype,
            )
            self.gamma2 = self.add_weight(
                name="gamma2",
                shape=[self.dim],
                initializer=tf.keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

        self.num_windows = int(
            (self.input_resolution // self.window_size)
            * (self.input_resolution // self.window_size)
        )
        super().build(input_shape)

    def call(self, inputs, q_global, training=None, *args, **kwargs):
        input_shape = tf.shape(inputs)
        H = input_shape[1]
        W = input_shape[2]
        C = input_shape[3]
        shortcut = inputs
        x = self.norm1(inputs)
        x_windows = window_partition(x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, C))
        attn_windows = self.attn(x_windows, q_global)
        x = window_reverse(attn_windows, self.window_size, H, W)
        x = shortcut + self._drop_path(inputs=self.gamma1 * x, training=training)
        x = x + self._drop_path(
            inputs=self.gamma2 * self.mlp(self.norm2(x)), training=training
        )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "input_resolution": self.input_resolution,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "drop": self.drop,
                "attn_drop": self.attn_drop,
                "drop_path": self.drop_path,
                "act_layer": self.act_layer,
                "attention": self.attention,
                "layer_scale": self.layer_scale,
            }
        )
        return config
