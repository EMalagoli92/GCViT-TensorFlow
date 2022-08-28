from typing import TypeVar, Type
import tensorflow as tf
import tensorflow_addons as tfa
from models.layers.utils import LayerNorm_, Conv2d_
from models.layers.se import SE
from models.utils import _to_channel_first, _to_channel_last

L = TypeVar("L",bound=tf.keras.layers.Layer)


@tf.keras.utils.register_keras_serializable(package='gcvit')
class ReduceSize(tf.keras.layers.Layer):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """
    def __init__(self,
                 dim: int,
                 norm_layer: Type[L] = LayerNorm_,
                 keep_dim: bool = False,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        dim : int
            Feature size dimension.
        norm_layer : tf.keras.layers.Layer, optional
            Normalization layer. 
            The default is LayerNorm_.
        keep_dim : bool, optional
            Bool argument for maintaining the resolution. 
            The default is False.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.norm_layer = norm_layer
        self.keep_dim = keep_dim
        
    def build(self,input_shape):
        self.conv1 = Conv2d_(in_channels = self.dim,
                             out_channels = self.dim,
                             kernel_size = 3,
                             stride = 1,
                             padding = 1,
                             groups = self.dim,
                             bias = False,
                             name = "conv/0"
                             )
        self.act1 = tfa.layers.GELU(approximate = False,
                                    name = "conv/1"
                                    )
        self.se = SE(inp = self.dim,
                     oup = self.dim,
                     name = "conv/2"
                     )
        self.conv2 = Conv2d_(in_channels = self.dim,
                             out_channels = self.dim,
                             kernel_size = 1,
                             stride = 1,
                             padding = 0,
                             bias = False,
                             name = "conv/3"
                             )
        if self.keep_dim:
            dim_out = self.dim
        else:
            dim_out = 2*self.dim
        self.reduction = Conv2d_(in_channels = self.dim,
                                 out_channels = dim_out,
                                 kernel_size = 3,
                                 stride = 2,
                                 padding = 1,
                                 bias = False,
                                 name = "reduction"
                                 )
        self.norm2 = self.norm_layer(normalized_shape = dim_out,
                                     name = "norm2"
                                     )
        self.norm1 = self.norm_layer(normalized_shape = self.dim,
                                     name = "norm1"
                                     )
        super().build(input_shape)
        
    def call(self,inputs,**kwargs):
        x = self.norm1(inputs)
        x = _to_channel_first(x)
        y = self.conv1(x)
        y = self.act1(y)
        y = self.se(y)
        y = self.conv2(y)
        x = x + y
        x = self.reduction(x)
        x = _to_channel_last(x)
        x = self.norm2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim,
                       "norm_layer": self.norm_layer,
                       "keep_dim": self.keep_dim
                       })
        return config