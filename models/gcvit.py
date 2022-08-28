import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from typing import TypeVar, Type, Optional, Union, List
import random
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_addons as tfa 
import tensorflow.experimental.numpy as tnp
from tensorflow.python.framework import random_seed
from models.config import MODELS_CONFIG, TAG, TF_WEIGHTS_URL
from models.utils import flatten_, _to_channel_first
from models.layers.utils import Dense_, LayerNorm_, Identity_
from models.layers.patch_embed import PatchEmbed
from models.layers.gcvit_layer import GCViTLayer

tnp.experimental_enable_numpy_behavior()

# Set Seed
SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
random_seed.set_seed(SEED)
np.random.seed(SEED)


L = TypeVar("L",bound=tf.keras.layers.Layer)


class GCViT_(tf.keras.Model):
    """
    GCViT based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """
    def __init__(self,
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
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 norm_layer: Type[L] = LayerNorm_,
                 layer_scale: Union[int,float,None] = None,
                 classifier_activation: Optional[str] = None,
                 data_format: Optional[str] = None,
                 **kwargs
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
            Dropout rate. The default is 0..
        attn_drop_rate : float, optional
            Attention dropout rate. 
            The default is 0.
        norm_layer : tf.keras.layers.Layer, optional
            Normalization layer. 
            The default is LayerNorm_.
        layer_scale : int | float, optional
            Layer scaling coefficient. 
            The default is None.
        classifier_activation: str, optional
            String name for a tf.keras.layers.Activation layer.
            The default is None.
        data_format: str, optional
            A string, one of channels_last (default) or channels_first. 
            The ordering of the dimensions in the inputs. channels_last 
            corresponds to inputs with shape 
            (batch_size, height, width, channels) 
            while channels_first corresponds to inputs with shape 
            (batch_size, channels, height, width).
            The default is None.
        """
        super().__init__(**kwargs)
        self.data_format = data_format
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, 
                                      dim=dim,
                                      name = "patch_embed"
                                      )
        self.pos_drop = tf.keras.layers.Dropout(rate = drop_rate,
                                                name = "pos_drop"
                                                )
        dpr = [x.numpy() for x in tf.linspace(0., drop_path_rate, sum(depths))]
        self.levels = []
        for i in range(len(depths)):
            level = GCViTLayer(dim = int(dim * 2 ** i),
                               depth = depths[i],
                               num_heads = num_heads[i],
                               window_size = window_size[i],
                               mlp_ratio = mlp_ratio,
                               qkv_bias = qkv_bias,
                               qk_scale = qk_scale,
                               drop=drop_rate, 
                               attn_drop = attn_drop_rate,
                               drop_path = dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer = norm_layer,
                               downsample = (i < len(depths) -1),
                               layer_scale = layer_scale,
                               input_resolution = int(2 ** (-2 -i) * resolution),
                               name = f"levels/{i}"
                               )
            self.levels.append(level)
        self.norm = norm_layer(num_features, 
                               name = "norm"
                               )
        self.avgpool = tfa.layers.AdaptiveAveragePooling2D(output_size = 1,
                                                           data_format="channels_first",
                                                           name = "avgpool"
                                                           )
        self.head = Dense_(num_features, num_classes, name = "head") if num_classes > 0 else Identity_(name = "head")
        if classifier_activation is not None:
            self.classifier_activation = tf.keras.layers.Activation(classifier_activation,
                                                                    dtype = self.dtype,
                                                                    name = "pred"
                                                                    )
        self.__set_data_format()
        
    def __set_data_format(self):
        if self.data_format is None:
            self.data_format = tf.keras.backend.image_data_format()
                
    def call_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for level in self.levels:
            x = level(x)
            
        x = self.norm(x)
        x = _to_channel_first(x)
        x = self.avgpool(x)
        x = flatten_(x, 1)
        return x
    
    def call(self,inputs,**kwargs):
        if self.data_format == "channels_last":
            inputs = _to_channel_first(inputs)
        x = self.call_features(inputs)
        x = self.head(x)
        if hasattr(self, "classifier_activation"):
            x = self.classifier_activation(x)
        return x
    
    
def GCViT(configuration: Optional[str] = None,
          pretrained: bool = False,
          **kwargs
          ) -> tf.keras.Model:
    '''
    Wrapper function for GCViT model.

    Parameters
    ----------
    configuration : Optional[str], optional
        Name of GCViT predefined configuration. 
        Possible values are: xxtiny, xtiny, tiny, small, base
        The default is None.
    pretrained : bool, optional
        Whether to use ImageNet pretrained weights.
        The default is False.

    Raises
    ------
    KeyError
        If choosen configuration not in:
        ['xxtiny','xtiny','tiny','small','base']

    Returns
    -------
    GCViT model (tf.keras.Model).
    '''
    if configuration is not None:
        if configuration in MODELS_CONFIG.keys():
            model =  GCViT_(**MODELS_CONFIG[configuration],
                            **kwargs
                            )
            if pretrained:
                if model.data_format == "channels_last":
                    model(tf.ones((1,224,224,3)))
                elif model.data_format == "channels_first":
                    model(tf.ones((1,3,224,224)))
                weights_path = "{}/{}/{}.h5".format(TF_WEIGHTS_URL,TAG,configuration)
                model_weights = tf.keras.utils.get_file(fname = "{}.h5".format(configuration),
                                                        origin = weights_path,
                                                        cache_subdir = "datasets/gcvit_tensorflow"
                                                        )
                model.load_weights(model_weights)
            return model
        else:
            raise KeyError(f"{configuration} configuration not found. Valid values are: xxtiny, xtiny, tiny, small, base.")
    else:
        return GCViT_(**kwargs)