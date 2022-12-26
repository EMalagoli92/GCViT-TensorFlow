from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa

from gcvit_tensorflow.models.layers.utils import Linear_, TruncNormalInitializer_


@tf.keras.utils.register_keras_serializable(package="gcvit")
class Mlp(tf.keras.layers.Layer):
    """Multi-Layer Perceptron (MLP) block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = "gelu",
        drop: float = 0.0,
        **kwargs
    ):
        """
        Parameters
        ----------
        in_features : int
            Input features dimension.
        hidden_features : Optional[int], optional
            Hidden features dimension.
            The default is None.
        out_features : Optional[int], optional
            Output features dimension.
            The default is None.
        act_layer : str, optional
            Name of activation layer.
            The default is "gelu".
        drop : float, optional
            Dropout rate.
            The default is 0.0.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.drop = drop

    def build(self, input_shape):
        self._out_features = self.out_features or self.in_features
        self._hidden_features = self.hidden_features or self.in_features
        self.fc1 = Linear_(
            in_features=self.in_features,
            units=self._hidden_features,
            kernel_initializer=TruncNormalInitializer_(std=0.02),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="fc1",
        )
        if self.act_layer == "gelu":
            self.act = tfa.layers.GELU(approximate=False, name="act")
        else:
            self.act = tf.keras.layers.Activation(
                self.act_layer, dtype=self.dtype, name="act"
            )
        self.fc2 = Linear_(
            in_features=self._hidden_features,
            units=self._out_features,
            kernel_initializer=TruncNormalInitializer_(std=0.02),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="fc2",
        )
        self._drop = tf.keras.layers.Dropout(rate=self.drop, name="drop")
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self._drop(x)
        x = self.fc2(x)
        x = self._drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "act_layer": self.act_layer,
                "drop": self.drop,
            }
        )
        return config
