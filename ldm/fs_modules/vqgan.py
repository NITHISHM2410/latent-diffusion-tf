import tensorflow as tf
from ldm.diffusion.diffusion_tf import Encoder, Decoder


class LPIPS:
    def __init__(self, lpips_path):
        self.model = tf.saved_model.load(lpips_path)

    def __call__(self, x1, x2):
        x1 = tf.transpose(x1, (0, 3, 1, 2))
        x2 = tf.transpose(x2, (0, 3, 1, 2))
        x1 = x1 * 2 - 1
        x2 = x2 * 2 - 1
        output = self.model(onnx__Sub_0=x1, onnx__Sub_1=x2)['198']
        return output


class Discriminator(tf.keras.Model):
    def __init__(self, c_in=3, ch=64, n_layers=3, hw=256, disc_weight=1.0):
        super(Discriminator, self).__init__()
        self.disc_weight = disc_weight

        kernel_size = 4
        padding = 'same'

        self.block = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(ch, kernel_size=kernel_size, strides=2, padding=padding),
            tf.keras.layers.LeakyReLU(0.2)
        ])

        for n in range(1, n_layers):
            block_out = min(2 ** n, 8)

            self.block.add(tf.keras.layers.SeparableConv2D(ch * block_out, kernel_size=kernel_size, strides=2,
                                                           padding=padding, use_bias=False))
            self.block.add(tf.keras.layers.GroupNormalization())
            self.block.add(tf.keras.layers.LeakyReLU(0.2))

        block_out = min(2 ** n_layers, 8)
        self.block.add(tf.keras.layers.SeparableConv2D(ch * block_out, kernel_size=kernel_size,
                                                       strides=1, padding=padding, use_bias=False))
        self.block.add(tf.keras.layers.GroupNormalization())
        self.block.add(tf.keras.layers.LeakyReLU(0.2))
        self.block.add(tf.keras.layers.SeparableConv2D(1, kernel_size=kernel_size, strides=1, padding=padding))

        self.block.build((None, None, None, c_in))

    def call(self, inputs, training=True, **kwargs):
        x = inputs
        x = self.block(x, training=training)
        return x

    def calculate_adaptive_weights(self, recons_loss, gen_loss, last_layer):
        recons_grads = tf.gradients(recons_loss, last_layer)[0]
        gen_grads = tf.gradients(gen_loss, last_layer)[0]

        d_weight = tf.norm(recons_grads) / (tf.norm(gen_grads) + 1e-4)
        d_weight = tf.clip_by_value(d_weight, 0.0, 1e4)
        d_weight = tf.stop_gradient(d_weight) * self.disc_weight
        return d_weight


class VQLayer(tf.keras.layers.Layer):
    def __init__(self, img_res, latent_dim, latent_vectors, beta):
        super(VQLayer, self).__init__()
        self.latent_dim = latent_dim
        self.img_res = img_res
        self.beta = beta

        self.embedding = tf.keras.layers.Embedding(latent_vectors, latent_dim)
        self.embedding.build((None, 1))

    def call(self, inputs, **kwargs):
        z = inputs
        ih, iw = tf.shape(z)[1], tf.shape(z)[2]
        z_flatten = tf.reshape(z, (-1, ih*iw, self.latent_dim))

        distances = (
                tf.math.reduce_sum(z_flatten ** 2, axis=-1, keepdims=True) +
                tf.math.reduce_sum(self.embedding.weights[0] ** 2, axis=-1) -
                2 * tf.matmul(z_flatten, self.embedding.weights[0], transpose_b=True)
        )
        min_ind = tf.argmin(distances, axis=-1)
        zq = self.embedding(min_ind)
        zq = tf.reshape(zq, (-1, ih, iw, self.latent_dim))

        loss = self.beta * tf.reduce_mean((tf.stop_gradient(zq) - z) ** 2) + tf.reduce_mean(
            (zq - tf.stop_gradient(z)) ** 2)
        zq = z + tf.stop_gradient(zq - z)
        return zq, min_ind, loss


class VQGAN(tf.keras.Model):
    def __init__(self, ch_list, attn_res, latent_vectors, latent_dim, vq_beta, img_size, c_in, c_out, zc):
        super(VQGAN, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.latent_vectors = latent_vectors

        self.encoder = Encoder(ch_list=ch_list, attn_res=attn_res,
                               img_res=img_size, c_in=c_in, c_out=zc, heads=1)
        self.decoder = Decoder(ch_list=ch_list, attn_res=attn_res,
                               img_res=img_size // (2 ** (len(ch_list) - 1)),
                               c_in=zc, c_out=c_out, heads=1)

        self.q_conv = tf.keras.layers.Conv2D(filters=latent_dim, kernel_size=1)
        self.post_q_conv = tf.keras.layers.Conv2D(filters=zc, kernel_size=1)

        self.vq_layer = VQLayer(img_res=img_size // (2 ** (len(ch_list) - 1)), beta=vq_beta,
                                latent_dim=latent_dim, latent_vectors=latent_vectors)

        self.build((None, None, None, self.c_in))

    def encode(self, x, training):
        x = self.encoder(inputs=x, training=training)
        x = self.q_conv(x)
        zq, min_ind, loss = self.vq_layer(x)
        return zq, min_ind, loss

    def decode(self, x, training):
        x = self.post_q_conv(x)
        x = self.decoder(inputs=x, training=training)
        x = tf.nn.sigmoid(x)
        return x

    # FOR LDM
    def encode_ldm(self, x, training=False):
        x = self.encoder(inputs=x, training=training)
        x = self.q_conv(x)
        return x

    # FOR LDM
    def decode_ldm(self, x, training=False):
        zq, min_ind, _ = self.vq_layer(x)
        x = self.post_q_conv(zq)
        x = self.decoder(inputs=x, training=training)
        x = tf.nn.sigmoid(x)
        return x

    def get_last_layer(self):
        return self.decoder.layers[-1].weights

    def call(self, inputs, training=True, **kwargs):
        x = inputs

        zq, min_ind, codebook_loss = self.encode(x, training)
        y = self.decode(zq, training)

        return y, codebook_loss

