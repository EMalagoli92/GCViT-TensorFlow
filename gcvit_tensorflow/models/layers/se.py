import tensorflow as tf
import tensorflow_addons as tfa

from gcvit_tensorflow.models.layers.utils import Linear_, TruncNormalInitializer_


@tf.keras.utils.register_keras_serializable(package="gcvit")
class SE(tf.keras.layers.Layer):
    """Squeeze and excitation block."""

    def __init__(self, inp: int, oup: int, expansion: float = 0.25, **kwargs):
        """
        Parameters
        ----------
        inp : int
            Input features dimension.
        oup : int
            Output features dimension.
        expansion : float, optional
            Expansion ratio.
            The default is 0.25.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.inp = inp
        self.oup = oup
        self.expansion = expansion

    def build(self, input_shape):
        self.avg_pool = tfa.layers.AdaptiveAveragePooling2D(
            output_size=1, data_format="channels_first", name="avg_pool"
        )
        self.fc1 = Linear_(
            in_features=self.oup,
            units=int(self.inp * self.expansion),
            use_bias=False,
            kernel_initializer=TruncNormalInitializer_(std=0.02),
            name="fc/0",
        )
        self.act1 = tfa.layers.GELU(approximate=False, name="fc1/1")
        self.fc2 = Linear_(
            in_features=int(self.inp * self.expansion),
            units=self.oup,
            use_bias=False,
            kernel_initializer=TruncNormalInitializer_(std=0.02),
            name="fc/2",
        )
        self.act2 = tf.keras.layers.Activation(activation=tf.nn.sigmoid, name="fc/3")
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        input_shape = tf.shape(inputs)
        b = input_shape[0]
        c = input_shape[1]
        y = self.avg_pool(inputs)
        y = tf.reshape(y, (b, c))
        y = self.fc1(y)
        y = self.act1(y)
        y = self.fc2(y)
        y = self.act2(y)
        y = tf.reshape(y, (b, c, 1, 1))
        return inputs * y

    def get_config(self):
        config = super().get_config()
        config.update({"inp": self.inp, "oup": self.oup, "expansion": self.expansion})
        return config
