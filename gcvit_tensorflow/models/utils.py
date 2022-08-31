import tensorflow as tf

      
def flatten_(x: tf.Tensor,
             start_dim: int
             ) -> tf.Tensor:
    '''
    TF2/Keras implementation of torch.flatten with end_dim = -1.

    Parameters
    ----------
    x : tf.Tensor
        Input Tensor.
    start_dim : int
        the first dim to flatten.

    Returns
    -------
    Flatten Tensor
    '''
    return tf.reshape(x,shape = (*x.shape[:start_dim],-1))


def _to_channel_last(x: tf.Tensor) -> tf.Tensor:
    """
    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape: (B, C, H, W)

    Returns
    -------
    tf.Tensor of shape: (B, H, W, C)
    """
    return tf.transpose(x, perm = [0, 2, 3, 1])


def _to_channel_first(x: tf.Tensor) -> tf.Tensor:
    """
    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape: (B, H, W, C)

    Returns
    -------
    tf.Tensor of shape: (B, C, H, W)
    """
    return tf.transpose(x,perm=[0, 3, 1, 2])


def window_partition(x: tf.Tensor, 
                     window_size: int
                     ) -> tf.Tensor:
    """
    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape: (B, H, W, C)
    window_size : int
        Window size.

    Returns
    -------
    windows : tf.Tensor
        Local window features (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = tf.reshape(x,(B, H // window_size, window_size, W // window_size, window_size, C))
    windows = tf.transpose(x,perm = [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows,(-1, window_size, window_size, C))
    return windows


def window_reverse(windows: tf.Tensor,
                   window_size: int,
                   H: int,
                   W: int
                   ) -> tf.Tensor:
    """
    Parameters
    ----------
    windows : tf.Tensor
        Local window features (num_windows*B, window_size, window_size, C)
    window_size : int
        Window size.
    H : int
        Height of image.
    W : int
        Width of image.

    Returns
    -------
    x : tf.Tensor
        Tensor of shape: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows,(B, H // window_size, W // window_size, window_size, window_size, -1))
    x = tf.transpose(x, perm = [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x,(B, H, W, -1))
    return x