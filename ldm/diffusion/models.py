import tensorflow as tf
import tensorflow_hub as hub
from ldm.diffusion.diffusion_tf import DiffusionUNet


class UnCondUNet(DiffusionUNet):
    def __init__(self, **kwargs):
        """
        An Uncond diffusion model.

        """
        super(UnCondUNet, self).__init__(**kwargs['base'])
        self.build_model()

    def apply_conditioning(self, conditions):
        conditions['img_org'] = conditions['img_cond'] = conditions['text_cond'] = conditions['class_cond'] = \
        conditions['cond_weight'] = None
        return conditions


class ClassCondUNet(DiffusionUNet):
    def __init__(self, num_classes, **kwargs):
        """

        :param num_classes: No of classes.

        Conditional input :-
            'class_cond': should be a tensor with shape (B, 1), where B is the batch size and 2nd dim is the label value.

         Requires a param 'cond_weight' in 'conditions' dict.
            'cond_weight': interpolation weight from uncond to cond sampling.
        """
        super(ClassCondUNet, self).__init__(**kwargs['base'])
        self.num_classes = num_classes
        self.cls_encoding = tf.keras.layers.Embedding(self.num_classes + 1, self.ch_list[0] * 4)
        self.flatten = tf.keras.layers.Reshape(target_shape=())

        self.build_model()

    def encode_inputs(self, inputs):
        inputs = super().encode_inputs(inputs.copy())
        inputs['class_cond'] = self.cls_encoding(self.flatten(inputs['class_cond']))
        inputs['time'] += inputs['class_cond']
        del inputs['class_cond']
        return inputs

    def apply_conditioning(self, conditions):
        conditions['img_org'] = conditions['img_cond'] = conditions['text_cond'] = None
        assert 'class_cond' in conditions and conditions[
            'class_cond'] is not None, "Provide class_cond for class conditioning model."
        conditions['cond_weight'] = tf.constant(conditions.get('cond_weight', self.cond_weight), dtype=tf.float32)
        return conditions


class UnMaskUNet(DiffusionUNet):
    def __init__(self, low_x_res, mask_edge_percent, min_max_mask_box, **kwargs):
        """

        :param low_x_res: x, where the image is down sampled x times post applying conditions and fed to the nn.
        if input img size is 256 and this param is 4 and then img size to nn is 64.
        :param mask_edge_percent: percentage of masking to be done from all sides of the image as a range (min, max).
        :param min_max_mask_box: min height, min width , max height, max width aof the mask box as list.

        Conditional input 'img_cond' must be in the below format:

            Dict:
                'image': image to be masked.
                'mask_edge_percent': tensor of shape (4, ) each denotes the mask percent from each edge. (Top, Down, Left, Right)
                'min_max_mask_box' : tensor of shape (4, ) which denotes the mask box location. format: (x, y, x+h, y+w).

            or

            Tensor:
                   Tensor of images. 'mask_edge_percent' and 'min_max_mask_box' is automatically set to None.

            'mask_edge_percent' and 'min_mask_box_hw' will be randomly assigned within the range mentioned in this class
            init, if set to None.

        """
        super(UnMaskUNet, self).__init__(**kwargs['base'])
        self.mask_edge_percent = mask_edge_percent
        self.min_max_mask_box = min_max_mask_box
        self.low_x_res = low_x_res

        self.build_model()

    @tf.function
    def mask_out(self, image, edge_mask_percent=None, mask_boxes=None):
        x = image
        if len(x.shape) == 3:
            x = x[None, :, :, :]

        minn, maxx = self.mask_edge_percent[0], self.mask_edge_percent[1]
        minn_h_box, minn_w_box = self.min_max_mask_box[0], self.min_max_mask_box[1]
        maxx_h_box, maxx_w_box = self.min_max_mask_box[2], self.min_max_mask_box[3]

        B, H, W, C = x.shape
        hh, ww = tf.meshgrid(tf.range(H) + 1, tf.range(W) + 1, indexing='ij')
        hh, ww = hh[None, :, :, None], ww[None, :, :, None]

        m = tf.random.uniform((B, 4), minn, maxx) if edge_mask_percent is None else edge_mask_percent
        m = tf.cast(m * tf.constant([H, H, W, W], dtype=tf.float32)[None, :], tf.int32)
        m = tf.transpose(tf.reshape(m, (B, 4, 1, 1, 1)), (1, 0, 2, 3, 4))
        x = tf.cast(~((hh < m[0]) | (hh > H - m[1]) | (ww < m[2]) | (ww > W - m[3])), tf.float32) * x

        if mask_boxes is None:
            mh = tf.random.uniform((B, 1, 1, 1), 0, H - minn_h_box, tf.int32)
            mhx = tf.clip_by_value(tf.random.uniform((B, 1, 1, 1), tf.reduce_min(mh), H, tf.int32), mh + minn_h_box,
                                   mh + maxx_h_box)
            mw = tf.random.uniform((B, 1, 1, 1), 0, W - minn_w_box, tf.int32)
            mwx = tf.clip_by_value(tf.random.uniform((B, 1, 1, 1), tf.reduce_min(mw), W, tf.int32), mw + minn_w_box,
                                   mw + maxx_w_box)
        else:
            mh, mw, mhx, mwx = tf.split(mask_boxes, 4, 1)
            mh, mw, mhx, mwx = mh[:, :, None, None], mw[:, :, None, None], mhx[:, :, None, None], mwx[:, :, None, None]

        x = tf.cast(~((hh > mh) & (hh < mhx) & (ww > mw) & (ww < mwx)), tf.float32) * x
        x = tf.image.resize(x, (H // self.low_x_res, W // self.low_x_res), 'bilinear')
        if B == 1:
            return x[0]
        return x

    def apply_conditioning(self, conditions):
        conditions['text_cond'] = conditions['class_cond'] = conditions['cond_weight'] = None
        assert 'img_cond' in conditions and conditions[
            'img_cond'] is not None, "Provide img_cond for class conditioning model."

        if isinstance(conditions['img_cond'], dict):
            conditions['img_org'] = conditions['img_cond']['image']
            conditions['img_cond'] = self.mask_out(**conditions['img_cond'])
        else:
            conditions['img_org'] = conditions['img_cond']
            conditions['img_cond'] = self.mask_out(conditions['img_cond'])

        return conditions


class TextCondUNet(DiffusionUNet):
    def __init__(self, model_path, **kwargs):
        """

        :param model_path: path of pretrained text encoder model.

        Conditional input :-
            'text_cond':  tokenized text 'input_word_ids' using bert tokenizer. output shape of bert tokenizer 'input_word_ids' = (BS, 128)

             Requires a param 'cond_weight' in 'conditions' dict.
                'cond_weight': interpolation weight from uncond to cond sampling.
        """
        super(TextCondUNet, self).__init__(**kwargs['base'])
        self.text_seq_len = 128

        self.bert_encoder = hub.KerasLayer(model_path)
        self.bert_encoder.trainable = False

        self.build_model(text_seq_dim=self.text_seq_len)

    def encode_inputs(self, inputs):
        inputs = super().encode_inputs(inputs.copy())
        tokens = tf.cast(inputs['text_cond'], tf.int32)

        inputs['text_cond'] = {
            'input_mask': tf.cast(tf.cast(tokens, tf.bool), tf.int32),
            'input_type_ids': tf.zeros_like(tokens),
            'input_word_ids': tokens
        }
        inputs['text_cond'] = self.bert_encoder(inputs['text_cond'])['sequence_output']
        return inputs

    def apply_conditioning(self, conditions):
        conditions['img_org'] = conditions['img_cond'] = conditions['class_cond'] = None
        assert 'text_cond' in conditions and conditions[
            'text_cond'] is not None, "Provide text_cond for text conditioning model."
        conditions['cond_weight'] = tf.constant(conditions.get('cond_weight', self.cond_weight), dtype=tf.float32)
        return conditions
