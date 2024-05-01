import yaml
import importlib
import tensorflow as tf
from PIL import Image


def read_images(image_locations, size, return_ds=False):
    image_locations = tf.data.Dataset.from_tensor_slices(image_locations)

    def transform(img):
        img = tf.io.decode_jpeg(tf.io.read_file(img), channels=3)
        img = tf.image.resize(img, (size, size))
        img = img / 255
        return img

    images = image_locations.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    if return_ds:
        return images

    images = images.batch(len(image_locations), drop_remainder=True)
    return next(iter(images))


def map_with_ds(x, map_fn, no_of):
    ds = tf.data.Dataset.from_tensor_slices(x).map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(no_of, True)
    return next(iter(ds))


def dist_with_dataset(device, x, no_of):
    x = distribute_dataset(device, tf.data.Dataset.from_tensor_slices(x).batch(no_of, drop_remainder=True))
    x = next(iter(x))
    return x


def dist_with_value(device, x):
    x = device.experimental_distribute_values_from_function(lambda f: x)
    return x


def distribute_tensor(device, x, no_of):
    if x is None:
        return x
    if len(x.shape) == 0:
        return dist_with_value(device, x)
    return dist_with_dataset(device, x, no_of)


def distribute_dataset(device, ds):
    return device.experimental_distribute_dataset(ds)


def accumulate_dist(device, dist_values):
    if dist_values is None:
        return dist_values
    return tf.concat(device.experimental_local_results(dist_values), axis=0)


def load_params_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        model_file = yaml.safe_load(file)
    return model_file


def load_model_yaml(yaml_file):
    model_file = load_params_yaml(yaml_file)
    model_dir = model_file['model_dir']
    model_name = model_file['model_name']
    params = model_file['params']

    if 'gen' in params and 'disc' in params:
        params = params['gen']

    model = getattr(importlib.import_module(model_dir), model_name)
    model = model(**params)

    return model


def img_hor_concat(img_sets, n, h, w):
    img_sets = [iter(im) for im in img_sets]
    images = []
    for _ in range(n):
        image_set = [next(i) for i in img_sets]
        image_set = [tf.cast(tf.image.resize(i[None, ...], (h, w)), tf.uint8) if i.shape[:2] != (h, w) else i[None, ...]
                     for i in image_set]
        merged_images = tf.concat(image_set, axis=2)
        merged_images = tf.pad(merged_images, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], constant_values=255)
        images.append(merged_images)
    images = tf.concat(images, axis=0)
    return images


def save_grid(img_sets, save_loc, n_cols=4, h=256, w=256):
    """

    :param img_sets: list of image sets. Images across different sets would be concatenated.
    :param save_loc: path, to save the image grid plot.
    :param n_cols: number of columns in grid plot.
    :param h: All images will be resized to have height h.
    :param w: All images will be resized to have width w.
    :return:

    """
    img_sets = [im for im in img_sets if im is not None]
    img_sets = [tf.cast(im * 255, tf.uint8) for im in img_sets]
    n = len(img_sets[0])

    if len(img_sets) > 1:
        images = iter(img_hor_concat(img_sets, n, h, w))
    else:
        images = iter(img_sets[0])

    img_grid = []

    for r in range(n // n_cols):
        row_plot = []
        for c in range(n_cols):
            row_plot.append(next(images))
        img_grid.append(tf.concat(row_plot, axis=1))

    img_grid = tf.concat(img_grid, axis=0)
    img_grid = tf.pad(img_grid, constant_values=255, paddings=[[2, 2], [2, 2], [0, 0]])
    img_grid = img_grid.numpy()
    img_grid = Image.fromarray(img_grid)
    img_grid.save(str(save_loc + '.png'))
