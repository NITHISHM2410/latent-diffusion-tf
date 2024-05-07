import os

import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel(3)
from tqdm import tqdm
from glob import glob


def serialize(img, label):
    sample = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }
    sample = tf.train.Example(features=tf.train.Features(feature=sample))
    return sample.SerializeToString()


def create_records(ddf, op_path):
    ddf = ddf.sample(len(ddf))
    with (tf.io.TFRecordWriter(op_path, options=tf.io.TFRecordOptions(compression_type='GZIP', compression_level=9)) as writer):
        for img, label in tqdm(ddf.values, leave=True, position=0):
            img = open(img, 'rb').read()
            label = classes.index(label)
            label = tf.io.serialize_tensor(tf.constant(label)).numpy()
            serialized_sample = serialize(img, label)
            writer.write(serialized_sample)


########################################################################################################################
# MAKE CHANGES HERE
# Path to images folder containing folders of classes
img_files_path = r"F:\Python Works\PyCharm Projects\LDM\dataset\poo\flowers"

# Validation data percentage
val_percent = 0.1
########################################################################################################################

classes = os.listdir(img_files_path)
img_files = glob(str(img_files_path + r"\*\*"))

df = pd.DataFrame({'image': img_files})
df['label'] = df['image'].apply(lambda x: x.split("\\")[-2])

val_df = df.groupby('label').sample(frac=val_percent)
train_df = df.drop(val_df.index)


########################################################################################################################
# MAKE CHANGES HERE
# 2nd param: output path for TFR with name.

create_records(train_df, r"F:\Python Works\PyCharm Projects\LDM\dataset\flowers_tf_train")
create_records(val_df, r"F:\Python Works\PyCharm Projects\LDM\dataset\flowers_tf_val")
########################################################################################################################

