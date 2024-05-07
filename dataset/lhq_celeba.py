# script to produce tf records for lhq or celeb-a dataset
import os
import random
from tqdm import tqdm
from glob import glob


import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def serialize(img):
    sample = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
    }
    sample = tf.train.Example(features=tf.train.Features(feature=sample))
    return sample.SerializeToString()


def create_records(img_files, op_path):
    with (tf.io.TFRecordWriter(op_path,
                               options=tf.io.TFRecordOptions(compression_type='GZIP', compression_level=9)) as writer):
        for img in tqdm(img_files, leave=True, position=0):
            img = open(img, 'rb').read()
            img = tf.io.decode_jpeg(img, 3)
            img = tf.io.encode_jpeg(img).numpy()
            serialized_sample = serialize(img)
            writer.write(serialized_sample)


########################################################################################################################
# MAKE CHANGES HERE
# Path to images folder
img_files_path = r"F:\lhq_256"

# Validation data percentage
val_percent = 0.1
########################################################################################################################

img_files = glob(str(img_files_path+'/*'))
val_files = random.sample(img_files, int(val_percent*len(img_files)))
train_files = list(set(img_files) - set(val_files))

create_records(train_files, "TrainRecords")
create_records(val_files, "ValRecords")

# Parsing code is available in Demo Notebook.
