from typing import TypeVar, Type, Optional, Union
import tensorflow as tf
from models.layers.utils import LayerNorm_
from models.layers.gcvit_block import GCViTBlock
from models.layers.window_attention import WindowAttention
from models.layers.window_attention_global import WindowAttentionGlobal
from models.layers.reduce_size import ReduceSize
from models.layers.global_query_gen import GlobalQueryGen
from models.utils import _to_channel_first

L = TypeVar("L",bound=tf.keras.layers.Layer)


@tf.keras.utils.register_keras_serializable(package='gcvit')    
class GCViTLayer(tf.keras.layers.Layer):
    """
    GCViT layer based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """
    def __init__(self,
                 dim: int,
                 depth: int,
                 input_resolution: int,
                 num_heads: int,
                 window_size: int,
                 downsample: bool = True,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[bool] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_layer: Type[L] = LayerNorm_,
                 layer_scale: Union[int,float,None] = None,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        dim : int
            Feature size dimension.
        depth : int
            Number of layers in each stage.
        input_resolution : int
            Input image resolution.
        num_heads : int
            Number of heads in each stage.
        window_size : int
            Window size in each stage.
        downsample : bool, optional
            Bool argument for down-sampling. 
            The default is True.
        mlp_ratio : float, optional
            MLP ratio. 
            The default is 4..
        qkv_bias : bool, optional
            Bool argument for query, key, value learnable bias. 
            The default is True.
        qk_scale : bool, optional
            Bool argument to scaling query, key. 
            The default is None.
        drop : float, optional
            Dropout rate. 
            The default is 0.
        attn_drop : float, optional
            Attention dropout rate. 
            The default is 0.
        drop_path : float, optional
            Drop path rate. 
            The default is 0.
        norm_layer : tf.keras.layers.Layer, optional
            Normalization layer. 
            The default is LayerNorm_.
        layer_scale : int | float, optional
            Scaling coefficient. 
            The default is None.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.downsample = downsample
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.norm_layer = norm_layer
        self.layer_scale = layer_scale
        
    def build(self,input_shape):
        self.blocks = [GCViTBlock(dim = self.dim,
                                  num_heads = self.num_heads,
                                  window_size = self.window_size,
                                  mlp_ratio = self.mlp_ratio,
                                  qkv_bias = self.qkv_bias,
                                  qk_scale = self.qk_scale,
                                  attention = WindowAttention if (i % 2 == 0) else WindowAttentionGlobal,
                                  drop = self.drop,
                                  attn_drop = self.attn_drop,
                                  drop_path = self.drop_path[i] if isinstance(self.drop_path, list) else self.drop_path,
                                  norm_layer = self.norm_layer,
                                  layer_scale = self.layer_scale,
                                  input_resolution = self.input_resolution,
                                  name = f"blocks/{i}"
                                  ) for i in range(self.depth)]
        
        self.downsample = None if not self.downsample else ReduceSize(dim=self.dim, 
                                                                      norm_layer=self.norm_layer,
                                                                      name = "downsample"
                                                                      )
        self.q_global_gen = GlobalQueryGen(dim = self.dim, 
                                           input_resolution = self.input_resolution, 
                                           window_size = self.window_size, 
                                           num_heads = self.num_heads, 
                                           name = "q_global_gen"
                                           )
        super().build(input_shape)
        
    def call(self,inputs,**kwargs):
        q_global = self.q_global_gen(_to_channel_first(inputs))
        x = inputs
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x
        return self.downsample(x)    
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim,
                       "depth": self.depth,
                       "input_resolution": self.input_resolution,
                       "num_heads": self.num_heads,
                       "window_size": self.window_size,
                       "downsample": self.downsample,
                       "mlp_ratio": self.mlp_ratio,
                       "qkv_bias": self.qkv_bias,
                       "qk_scale": self.qk_scale,
                       "drop": self.drop,
                       "attn_drop": self.attn_drop,
                       "drop_path": self.drop_path,
                       "norm_layer": self.norm_layer,
                       "layer_scale": self.layer_scale
                       })
        return config