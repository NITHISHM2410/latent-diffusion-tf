import tensorflow as tf
from ldm.diffusion.transformer import SpatialTransformer


class Sequential(tf.keras.Sequential):
    """
    A Simple Sequential model made to handle kwargs and args during call.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    def call(self, inputs, training=True, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training, **kwargs)
        return x


class ForwardDiffusion:
    def __init__(self, time_steps, beta_start, beta_end):
        """
        Forward diffusion phase - q(xt | xt-1).

        :param time_steps:  diffusion time steps count.
        :param beta_start: variance schedule start.
        :param beta_end: variance schedule end.
        """

        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = tf.linspace(self.beta_start, self.beta_end, int(self.time_steps))
        self.alphas = 1. - self.betas
        self.alpha_hats = tf.math.cumprod(self.alphas)

    def __call__(self, image, time):
        x, t = image, time
        noise = tf.random.normal(shape=tf.shape(x))

        sqrt_alpha_hat = tf.math.sqrt(
            tf.gather(self.alpha_hats, t[:, -1])
        )[:, None, None, None]

        sqrt_one_minus_alpha_hat = tf.math.sqrt(
            1. - tf.gather(self.alpha_hats, t[:, -1])
        )[:, None, None, None]

        noised_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return noised_image, noise

    def get_forward_diffusion_params(self):
        return self.alphas, self.betas, self.alpha_hats


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed):
        """
         Positional embedding layer for embedding time.

        :param embed: embedding dim.
        """
        super(PositionalEmbedding, self).__init__()
        self.embed = embed

        self.emb_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed),
            tf.keras.layers.Activation("swish"),
            tf.keras.layers.Dense(self.embed)
        ])

    def call(self, inputs, **kwargs):
        t = inputs
        half_embed = (self.embed // 4) // 2
        emb = tf.math.log(10000.0) / (half_embed - 1)
        emb = tf.exp(tf.range(half_embed, dtype=tf.float32) * -emb)
        emb = tf.cast(t, tf.float32) * emb[None, :]
        emb = tf.concat([
            tf.sin(emb),
            tf.cos(emb)
        ], axis=-1)
        return self.emb_layer(emb)


class UpSample(tf.keras.layers.Layer):
    def __init__(self, c_out, with_conv):
        """
        Up sampling layer.

        :param c_out: expected output channels for this layer's outputs.
        :param with_conv: whether to use conv layer along with up sampling.
        """
        super(UpSample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = tf.keras.layers.SeparableConv2D(filters=c_out, kernel_size=3, padding='same')

    def call(self, inputs, **kwargs):
        x = inputs
        ih, iw = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.image.resize(x, size=(ih * 2, iw * 2), method='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class DownSample(tf.keras.layers.Layer):
    def __init__(self, c_out, with_conv):
        """
        Down sampling layer

        :param c_out: expected output channels for this layer's outputs.
        :param with_conv: whether to use conv layer for down sampling, 'False' equals pooling.
        """
        super(DownSample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = tf.keras.layers.SeparableConv2D(
                filters=c_out, kernel_size=3,
                strides=2, padding='same'
            )
        else:
            self.pool = tf.keras.layers.AvgPool2D(pool_size=(2, 2), padding='same')

    def call(self, inputs, **kwargs):
        x = inputs
        if self.with_conv:
            x = self.conv(x)
        else:
            x = self.pool(x)
        return x


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, c_in, c_out, dropout, t_emb, norm_g, img_cond=False):
        """
        Resnet block which implements basic convolutions and embeds time & optionally image_cond embedding to the input.

        :param c_in: input channels of this layer's inputs.
        :param c_out: expected output channels for this layer's outputs.
        :param dropout: dropout value.
        :param t_emb: embedding dimension of time embedding.
        :param norm_g: number of groups for group norm.
        :param img_cond: boolean value, whether layer must receive img_cond of then input (for img_cond task).
        """
        super(ResBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.t_emb = t_emb
        self.img_cond = img_cond

        self.norm1 = tf.keras.layers.GroupNormalization(norm_g)
        self.non_linear1 = tf.keras.layers.Activation("swish")
        self.conv1 = tf.keras.layers.SeparableConv2D(filters=self.c_out, kernel_size=3, padding='same')

        if self.t_emb is not None:
            self.time_emb = tf.keras.Sequential([
                tf.keras.layers.Activation("swish"),
                tf.keras.layers.Dense(self.c_out),
                tf.keras.layers.Reshape((1, 1, self.c_out))
            ])

        if self.img_cond:
            self.img_cond_emb = tf.keras.Sequential([
                tf.keras.layers.Activation("swish"),
                tf.keras.layers.Dense(c_out)
            ])

        self.norm2 = tf.keras.layers.GroupNormalization(groups=norm_g)
        self.non_linear2 = tf.keras.layers.Activation("swish")
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.conv2 = tf.keras.layers.SeparableConv2D(filters=self.c_out, kernel_size=3, padding='same')

        if self.c_in != self.c_out:
            self.conv_p = tf.keras.layers.Conv2D(filters=self.c_out, kernel_size=1)

    def call(self, inputs, **kwargs):
        x = inputs
        h, t, m = x, kwargs['time'], kwargs['img_cond']
        hr = h
        ih, iw = tf.shape(x)[1], tf.shape(x)[2]

        h = self.non_linear1(self.norm1(h))
        h = self.conv1(h)

        if self.t_emb is not None:
            h += self.time_emb(t)
        if self.img_cond:
            h += tf.image.resize(self.img_cond_emb(m), (ih, iw), method='bilinear')

        h = self.non_linear2(self.norm2(h))
        if kwargs['training']:
            h = self.dropout_layer(h)
        h = self.conv2(h)

        if self.c_in != self.c_out:
            hr = self.conv_p(hr)

        return h + hr


class AttentionUnitLayer(tf.keras.layers.Layer):
    def __init__(self, c, norm_g):
        """
         unit head work of multi head attention.

        :param c: input channels of this layer's inputs.
        :param norm_g: number of groups for group norm.
        """
        super(AttentionUnitLayer, self).__init__()
        self.c = c
        self.attn = tf.keras.layers.Attention(use_scale=True)
        self.norm = tf.keras.layers.GroupNormalization(groups=norm_g)
        self.qkv_proj = tf.keras.layers.Conv1D(kernel_size=1, filters=c * 3)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.norm(x)
        x = tf.split(self.qkv_proj(x), num_or_size_splits=3, axis=-1)
        x = self.attn(x)
        return x


class MHAAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, c, norm_g):
        """
         A Multi head attention layer.

        :param c: input channels of this layer's inputs.
        :param norm_g: number of groups for group norm.
        """
        super(MHAAttentionBlock, self).__init__()
        self.c = c
        self.attn_heads_units = [AttentionUnitLayer(self.c, norm_g)]
        self.final_proj = tf.keras.layers.Conv1D(kernel_size=1, filters=self.c)

    def call(self, inputs, **kwargs):
        x = inputs
        h = x
        ih, iw = tf.shape(x)[1], tf.shape(x)[2]
        h = tf.reshape(h, (-1, ih*iw, self.c))
        h = self.attn_heads_units[0](h)
        h = self.final_proj(h)
        h = tf.reshape(h, (-1, ih, iw, self.c))
        return h + x


class Encoder(tf.keras.Model):
    def __init__(self, c_in=3, c_out=512, ch_list=(128, 128, 256, 256, 512, 512), attn_res=(16,), norm_g=32,
                 resamp_with_conv=True, num_res_blocks=2, img_res=256, dropout=0):
        """
        An Image encoder.

        :param c_in: input channels of this model's inputs.
        :param c_out: output channels of this model's outputs.
        :param ch_list: list of channels to be used across down & up sampling.
        :param attn_res: list of resolution for which attention mechanism is to be implemented.
        :param norm_g: number of groups for group norm.
        :param resamp_with_conv: boolean value whether to use conv layer during up and down sampling.
        :param num_res_blocks: number of resnet blocks per channel in 'ch_list'.
        :param img_res: input image resolution.
        :param dropout: dropout value to be used in resnet blocks.
        """
        super(Encoder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.img_res = img_res
        num_res = len(ch_list)
        cur_res = self.img_res

        # down
        self.down_layers = [tf.keras.layers.SeparableConv2D(kernel_size=3, filters=ch_list[0], padding='same')]
        for level in range(num_res):
            block_in = ch_list[max(level - 1, 0)]
            block_out = ch_list[level]

            for block in range(num_res_blocks):
                ResAttnBlock = Sequential()
                ResAttnBlock.add(ResBlock(block_in, block_out, dropout, None, norm_g, False))
                block_in = block_out
                if cur_res in attn_res:
                    ResAttnBlock.add(MHAAttentionBlock(block_in, norm_g))
                self.down_layers.append(ResAttnBlock)

            if level != num_res - 1:
                self.down_layers.append(DownSample(ch_list[level], resamp_with_conv))
                cur_res //= 2

        # mid
        self.mid_layers = []
        self.mid_layers.append(ResBlock(ch_list[-1], ch_list[-1], dropout, None, norm_g, False))
        self.mid_layers.append(MHAAttentionBlock(ch_list[-1], norm_g))
        self.mid_layers.append(ResBlock(ch_list[-1], ch_list[-1], dropout, None, norm_g, False))

        # end
        self.end_norm = tf.keras.layers.GroupNormalization(groups=norm_g)
        self.end_non_linear = tf.keras.layers.Activation("swish")
        self.end_conv = tf.keras.layers.Conv2D(self.c_out, 3, padding='same')

        self.build((None, None, None, self.c_in))

    def call(self, inputs, training=True, **kwargs):
        x = inputs

        t = None
        ir = None

        x = self.down_layers[0](x)
        for layer in self.down_layers[1:]:
            x = layer(x, training=training, time=t, img_cond=ir)

        for layer in self.mid_layers:
            x = layer(x, time=t, img_cond=ir, training=training)

        x = self.end_conv(self.end_non_linear(self.end_norm(x)))
        return x


class Decoder(tf.keras.Model):
    def __init__(self, c_in=512, c_out=3, ch_list=(128, 128, 256, 256, 512, 512), attn_res=(16,), norm_g=32,
                 resamp_with_conv=True, num_res_blocks=2, img_res=256, dropout=0):
        """
        An Image Decoder.

        :param c_in: input channels of this model's inputs.
        :param c_out: output channels of this model's outputs.
        :param ch_list: list of channels to be used across down & up sampling.
        :param attn_res: list of resolution for which attention mechanism is to be implemented.
        :param norm_g: number of groups for group norm.
        :param resamp_with_conv: boolean value whether to use conv layer during up and down sampling.
        :param num_res_blocks: number of resnet blocks per channel in 'ch_list'.
        :param img_res: input image resolution.
        :param dropout: dropout value to be used in resnet blocks.
        """
        super(Decoder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.img_res = img_res
        num_res = len(ch_list)
        cur_res = self.img_res

        # up conv in
        self.conv_in = tf.keras.layers.SeparableConv2D(kernel_size=3, padding='same', filters=ch_list[-1])

        # mid
        self.mid_layers = []
        self.mid_layers.append(ResBlock(ch_list[-1], ch_list[-1], dropout, None, norm_g, False))
        self.mid_layers.append(MHAAttentionBlock(ch_list[-1], norm_g))
        self.mid_layers.append(ResBlock(ch_list[-1], ch_list[-1], dropout, None, norm_g, False))

        # up
        self.up_layers = []
        for level in reversed(range(num_res)):
            block_in = ch_list[min(level + 1, num_res - 1)]
            block_out = ch_list[level]

            for block in range(num_res_blocks + 1):
                ResAttnBlock = Sequential()
                ResAttnBlock.add(ResBlock(block_in, block_out, dropout, None, norm_g, False))
                block_in = block_out
                if cur_res in attn_res:
                    ResAttnBlock.add(MHAAttentionBlock(block_in, norm_g))
                self.up_layers.append(ResAttnBlock)
            if level != 0:
                self.up_layers.append(UpSample(ch_list[level], resamp_with_conv))
                cur_res *= 2

        # end
        self.end_norm = tf.keras.layers.GroupNormalization(norm_g)
        self.end_non_linear = tf.keras.layers.Activation("swish")
        self.end_conv = tf.keras.layers.Conv2D(self.c_out, 3, padding='same')

        self.build((None, None, None, self.c_in))

    def call(self, inputs, training=True, **kwargs):
        x = inputs

        t = None
        ir = None

        x = self.conv_in(x)
        for layer in self.mid_layers:
            x = layer(x, time=t, img_cond=ir, training=training)

        for layer in self.up_layers:
            x = layer(x, time=t, img_cond=ir, training=training)

        x = self.end_conv(self.end_non_linear(self.end_norm(x)))
        return x


class DiffusionUNet(tf.keras.Model):
    def __init__(self, c_in=3, c_out=3, ch_list=(128, 256, 256, 256), norm_g=32, attn_res=(16,),  sp_attn_depth=1,
                 trans_dim=128, resamp_with_conv=True, num_res_blocks=2, img_res=64, dropout=0, img_cond=False, text_cond=False,
                 class_cond=False, time_steps=1000, beta_start=1e-4, beta_end=0.02, cond_weight=3,
                 ):
        """
        An UNet model down samples and up samples and allows skip connections across both the up and down sampling.
        Also applies Forward diffusion, Positional embedding and class conditioning.

        :param c_in: input channels of this model's inputs.
        :param c_out: output channels of this model's outputs.
        :param ch_list: list of channels to be used across down & up sampling.
        :param norm_g: number of groups for group norm.
        :param attn_res: list of resolution for which attention mechanism is to be implemented.
        :param sp_attn_depth: spatial attention transformer depth.
        :param trans_dim: spatial attention transformer width.
        :param resamp_with_conv: boolean value whether to use conv layer during up and down sampling.
        :param num_res_blocks: number of resnet blocks per channel in 'ch_list'.
        :param img_res: input image resolution.
        :param dropout: dropout value to be used in resnet blocks.
        :param img_cond: boolean, whether to train model with image references for img conditional generation.
        :param text_cond: boolean, whether to train model with text conditioning.
        :param class_cond: boolean, whether to train model with class conditioning.
        :param time_steps: number of diffusion time steps.
        :param beta_start: noise variance schedule start value.
        :param beta_end: noise variance schedule end value.
        :param cond_weight: interpolation weight for unconditional-conditional generation.



        """
        super(DiffusionUNet, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.img_res = img_res
        self.ch_list = ch_list

        num_res = len(ch_list)
        cur_res = self.img_res

        self.class_cond = class_cond
        self.text_cond = text_cond
        self.img_cond = img_cond
        self.time_steps = time_steps

        # down
        self.down_layers = [tf.keras.layers.SeparableConv2D(kernel_size=3, filters=ch_list[0], padding='same')]
        self.skip_con_channels = [ch_list[0]]

        for level in range(num_res):
            block_in = ch_list[max(level - 1, 0)]
            block_out = ch_list[level]

            for block in range(num_res_blocks):
                ResAttnBlock = Sequential()
                ResAttnBlock.add(ResBlock(block_in, block_out, dropout, self.ch_list[0]*4, norm_g, img_cond))
                block_in = block_out
                if cur_res in attn_res:
                    ResAttnBlock.add(
                        MHAAttentionBlock(block_in, norm_g) if not text_cond else
                        SpatialTransformer(block_in, trans_dim, sp_attn_depth)
                    )

                self.down_layers.append(ResAttnBlock)
                self.skip_con_channels.append(block_in)
            if level != num_res - 1:
                self.down_layers.append(DownSample(ch_list[level], resamp_with_conv))
                cur_res //= 2
                self.skip_con_channels.append(ch_list[level])

        # mid
        self.mid_layers = []
        self.mid_layers.append(ResBlock(ch_list[-1], ch_list[-1], dropout, self.ch_list[0] * 4, norm_g, img_cond))

        self.mid_layers.append(MHAAttentionBlock(ch_list[-1], norm_g) if not text_cond else
                               SpatialTransformer(ch_list[-1], trans_dim, sp_attn_depth))

        self.mid_layers.append(ResBlock(ch_list[-1], ch_list[-1], dropout, self.ch_list[0] * 4, norm_g, img_cond))

        # up
        self.up_layers = []
        for level in reversed(range(num_res)):
            block_in = ch_list[min(level + 1, num_res - 1)]
            block_out = ch_list[level]

            for block in range(num_res_blocks + 1):
                ResAttnBlock = Sequential()
                skip_ch = block_in + self.skip_con_channels.pop()
                ResAttnBlock.add(ResBlock(skip_ch, block_out, dropout, self.ch_list[0] * 4, norm_g, img_cond))
                block_in = block_out
                if cur_res in attn_res:
                    ResAttnBlock.add(MHAAttentionBlock(block_in, norm_g) if not text_cond else
                                     SpatialTransformer(block_in, trans_dim, sp_attn_depth))
                self.up_layers.append(ResAttnBlock)
            if level != 0:
                self.up_layers.append(UpSample(ch_list[level], resamp_with_conv))
                cur_res *= 2

        # final
        self.exit_layers = tf.keras.Sequential([
            tf.keras.layers.GroupNormalization(groups=norm_g),
            tf.keras.layers.Activation("swish"),
            tf.keras.layers.SeparableConv2D(filters=c_out, kernel_size=3, padding='same')
        ])

        self.cond_weight = cond_weight

        self.pos_encoding = PositionalEmbedding(embed=self.ch_list[0] * 4)
        self.forward_diff = ForwardDiffusion(self.time_steps, beta_start=beta_start, beta_end=beta_end)
        self.alphas, self.betas, self.alpha_hats = self.forward_diff.get_forward_diffusion_params()

    def build_model(self, text_seq_dim=256):
        self.build({
            'batch': (None, None, None, self.c_in),
            'time': (None, 1),
            'img_cond': (None, None, None, self.c_in),
            'class_cond': (None, 1),
            'text_cond': (None, text_seq_dim)
        })

    def apply_conditioning(self, conditions):
        raise NotImplementedError

    def encode_inputs(self, inputs):
        inputs['time'] = self.pos_encoding(inputs['time'])
        return inputs

    def call(self, inputs, training=None, **kwargs):
        inputs = self.encode_inputs(inputs)
        x = inputs['batch']
        t = inputs['time']
        ir = inputs['img_cond']
        context = inputs['text_cond']

        x = self.down_layers[0](x)
        skip_cons = [x]
        for layer in self.down_layers[1:]:
            x = layer(x, training=training, time=t, img_cond=ir, context=context)
            skip_cons.append(x)

        for layer in self.mid_layers:
            x = layer(x, training=training, time=t, img_cond=ir, context=context)

        for layer in self.up_layers:
            if isinstance(layer, UpSample) or isinstance(layer, DownSample):
                x = layer(x, training=training, time=t, img_cond=ir, context=context)
            else:
                x = layer(tf.concat([x, skip_cons.pop()], axis=-1), training=training, time=t, img_cond=ir,
                          context=context)

        x = self.exit_layers(x)
        return x
