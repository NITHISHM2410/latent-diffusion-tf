import tensorflow as tf


class GEGLU(tf.keras.layers.Layer):
    def __init__(self, c_out):
        """
        A GeGLU Activation

        :param c_out: output channels for this layer.
        """
        super(GEGLU, self).__init__()
        self.dense = tf.keras.layers.Dense(c_out * 2)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x, y = tf.split(self.dense(x), num_or_size_splits=2, axis=-1)
        return x * tf.nn.gelu(y)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, c, multi=4):
        """
        A Feedforward layer for transformer based modules.

        :param c: number of input channels.
        :param multi: multiplier of input channels for hidden neurons.
        """

        super(FeedForward, self).__init__()
        self.block = tf.keras.Sequential([
            GEGLU(c * multi),
            tf.keras.layers.Dense(c)
        ])

    def call(self, inputs, training=True, *args, **kwargs):
        x = inputs
        return self.block(x, training=training)


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, c):
        """
        A Cross attention module for transformer

        :param c: number of input/output channels.

        """
        super(CrossAttention, self).__init__()
        self.attn = tf.keras.layers.Attention(use_scale=True)
        self.k = tf.keras.layers.Dense(c, use_bias=False)
        self.q = tf.keras.layers.Dense(c, use_bias=False)
        self.v = tf.keras.layers.Dense(c, use_bias=False)
        self.out = tf.keras.layers.Dense(c)

    def call(self, inputs, training=True, *args, **kwargs):
        x = inputs
        context = kwargs['context']
        if context is None:
            context = x
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)
        x = self.attn([q, v, k])
        x = self.out(x)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, c):
        """
        A Transformer block consisting of Feedforward and cross attn modules.

        :param c: number of input/output channels.

        """
        super(TransformerBlock, self).__init__()
        self.cross1 = CrossAttention(c=c)
        self.ff = FeedForward(c)
        self.cross2 = CrossAttention(c=c)

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=True, *args, **kwargs):
        x = inputs
        context = kwargs['context']
        x = self.cross1(self.norm1(x), context=None) + x
        x = self.cross2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, c, inner_dim, depth):
        """
        A transformer module consisting of multiple transformer blocks.

        :param c: number of input/output channels.
        :param inner_dim: transformer embed dim.
        :param depth: depth of transformer.
        """
        super(SpatialTransformer, self).__init__()
        self.inner_dim = inner_dim
        self.norm = tf.keras.layers.GroupNormalization(c)
        self.proj_in = tf.keras.layers.Conv2D(self.inner_dim, kernel_size=1, strides=1)

        self.blocks = [TransformerBlock(self.inner_dim) for _ in range(depth)]

        self.out = tf.keras.layers.Conv2D(c, kernel_size=1, strides=1)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        context = kwargs['context']
        x_in = x
        ih, iw = tf.shape(x)[1], tf.shape(x)[2]
        x = self.norm(x)
        x = self.proj_in(x)
        x = tf.reshape(x, (-1, ih * iw, self.inner_dim))
        for block in self.blocks:
            x = block(x, context=context)
        x = tf.reshape(x, (-1, ih, iw, self.inner_dim))
        x = self.out(x)
        return x + x_in
