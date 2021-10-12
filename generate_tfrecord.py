from TFRecord_utils import create_tfrdataset
# import tensorflow as tf
create_tfrdataset(PATH='dataset/train', tfrecord_file='dataset/train.tfrecord')
create_tfrdataset(PATH='dataset/test', tfrecord_file='dataset/test.tfrecord')