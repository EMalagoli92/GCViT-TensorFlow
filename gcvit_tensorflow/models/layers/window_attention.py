from typing import Optional
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from gcvit_tensorflow.models.layers.utils import Dense_, TruncNormalInitializer_
from gcvit_tensorflow.models.utils import flatten_


@tf.keras.utils.register_keras_serializable(package='gcvit')
class WindowAttention(tf.keras.layers.Layer):
    """
    Local window attention based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"    
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int,
                 qkv_bias: bool = True,
                 qk_scale: Optional[bool] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        dim : int
            Feature size dimension.
        num_heads : int
            Number of attention head.
        window_size : int
            Window size.
        qkv_bias : bool, optional
            Bool argument for query, key, value learnable bias. 
            The default is True.
        qk_scale : bool, optional
            Bool argument to scaling query, key. 
            The default is None.
        attn_drop : float, optional
            Attention dropout rate. 
            The default is 0.
        proj_drop : float, optional
            Output dropout rate. 
            The default is 0.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        
    def build(self,input_shape):
        self._window_size = (self.window_size,self.window_size)
        head_dim = self.dim // self.num_heads
        self.scale = self.qk_scale or head_dim ** -0.5
        
        coords_h = tnp.arange(self._window_size[0])
        coords_w = tnp.arange(self._window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w,indexing='ij'))
        coords_flatten = flatten_(coords,1)
        relative_coords = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        relative_coords = tf.transpose(relative_coords,perm = [1, 2, 0])
        relative_coords_np = relative_coords.numpy()
        relative_coords_np[:, :, 0] += self._window_size[0] - 1
        relative_coords_np[:, :, 1] += self._window_size[1] - 1
        relative_coords_np[:, :, 0] *= 2 * self._window_size[1] - 1
        relative_coords = tf.convert_to_tensor(relative_coords_np)
        relative_position_index = tf.math.reduce_sum(relative_coords,axis=-1)
        self.relative_position_index = self.add_weight(name = "relative_position_index",
                                                       shape = relative_position_index.shape,
                                                       initializer = tf.keras.initializers.Constant(relative_position_index),
                                                       trainable = False,
                                                       dtype = relative_position_index.dtype
                                                       )
        self.qkv = Dense_(in_features = self.dim,
                          out_features = self.dim *3,
                          bias = self.qkv_bias,
                          name = "qkv"
                          )
        self._attn_drop = tf.keras.layers.Dropout(rate = self.attn_drop,
                                                  name = "attn_drop"
                                                  )
        self.proj = Dense_(in_features = self.dim,
                           out_features = self.dim,
                           name = 'proj'
                           )
        self._proj_drop = tf.keras.layers.Dropout(rate = self.proj_drop,
                                                  name = "proj_drop"
                                                  )
        self.relative_position_bias_table = self.add_weight(name = "relative_position_bias_table",
                                                            shape = ((2 * self._window_size[0] - 1) * (2 * self._window_size[1] - 1), self.num_heads),
                                                            initializer = TruncNormalInitializer_(std = .02),
                                                            trainable = True,
                                                            dtype = self.dtype
                                                            )
        self.softmax = tf.keras.layers.Softmax(axis=-1,
                                               name = "softmax"
                                               )
        super().build(input_shape)
        
    def call(self,inputs, q_global,**kwargs):
        B_, N, C = inputs.shape
        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv,(B_, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm = [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ tnp.swapaxes(k,-2,-1))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index,[-1]))
        relative_position_bias = tf.reshape(relative_position_bias,(self._window_size[0] * self._window_size[1], self._window_size[0] * self._window_size[1], -1))
        relative_position_bias = tf.transpose(relative_position_bias,perm = [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias,axis=0)
        attn = self.softmax(attn)
        attn = self._attn_drop(attn)
        
        x = attn @ v
        x = tnp.swapaxes(x,1,2)
        x = tf.reshape(x,shape = (B_, N, C))
        x = self.proj(x)
        x = self._proj_drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"dim" : self.dim,
                       "num_heads": self.num_heads,
                       "window_size": self.window_size,
                       "qkv_bias": self.qkv_bias,
                       "qk_scale": self.qk_scale,
                       "attn_drop": self.attn_drop,
                       "proj_drop": self.proj_drop
                       })
        return config