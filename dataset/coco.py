# script to produce tf records for coco dataset
# below script creates image caption pair for future LDM txt2img model, so there may be same images with multiple captions.
# vq models are trained on coco dataset with duplicated images and captions dropped.


import tensorflow as tf

tf.get_logger().setLevel(3)

from tqdm import tqdm
import pandas as pd
import json
import os


def serialize(img, context):
    if context:
        sample = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'context': tf.train.Feature(bytes_list=tf.train.BytesList(value=[context]))
        }
    else:
        sample = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        }
    sample = tf.train.Example(features=tf.train.Features(feature=sample))
    return sample.SerializeToString()


def create_records(dataset: pd.DataFrame, dataset_path, op_path, for_ae):
    if for_ae:
        dataset = dataset.drop_duplicates(subset=('image_id',))
    with (tf.io.TFRecordWriter(op_path,
                               options=tf.io.TFRecordOptions(compression_type='GZIP', compression_level=9)) as writer):
        for img, context in tqdm(dataset.values, leave=True, position=0):
            img = open(os.path.join(dataset_path, img), 'rb').read()
            img = tf.io.decode_jpeg(img, 3)
            img = tf.cast(tf.image.resize(img, (256, 256)), tf.uint8)
            img = tf.io.encode_jpeg(img).numpy()
            if for_ae:
                context = context.encode('utf-8')
            else:
                context = None
            serialized_sample = serialize(img, context)
            writer.write(serialized_sample)


def load_data(annot_file):
    with open(annot_file) as f:
        df = json.load(f)['annotations']

    df = pd.DataFrame({'details': df})
    df['image_id'] = df['details'].apply(lambda x: x['image_id'])
    df['context'] = df['details'].apply(lambda x: x['caption'])
    df['image_id'] = df['image_id'].apply(lambda x: '%012d.jpg' % x)
    del df['details']
    return df


########################################################################################################################
# MAKE CHANGES HERE
# Change the path of annotations file with your annotation file path.

train_annot = r"F:\Python Works\PyCharm Projects\LDM\coco-2017-dataset\coco2017\annotations\captions_train2017.json"
val_annot = r"F:\Python Works\PyCharm Projects\LDM\coco-2017-dataset\coco2017\annotations\captions_val2017.json"

########################################################################################################################


train_df = load_data(train_annot)
val_df = load_data(val_annot)

########################################################################################################################
# MAKE CHANGES HERE
# Change the 2nd, 3rd, 4th param.
# 2nd param is the path of folder containing images. 3rd param is the path to generate TFR with name.
# 4th param: boolean. If True, TFR is created without captions, so all duplicates are dropped, only unique images are serialized.
#                     If False, TFR is created with captions, so there may be multiple copies of same image because same image may be multiple captions.

create_records(train_df, r"F:\Python Works\PyCharm Projects\LDM\coco-2017-dataset\coco2017\train2017", "TrainRecords", True)
create_records(val_df, r"F:\Python Works\PyCharm Projects\LDM\coco-2017-dataset\coco2017\val2017", "ValRecords", True)

########################################################################################################################


# Parsing code is available in Demo Notebook.
