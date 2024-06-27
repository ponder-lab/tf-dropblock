import tensorflow as tf
from nets.dropblock import DropBlock3D

tf.enable_eager_execution()

# only support `channels_last` data format
a = tf.ones([[1, 5, 5, 5, 1]])

drop_block = DropBlock3D(keep_prob=0.2, block_size=3)
b = drop_block(a, training=True)

print(b[0, :, :, 0])

# update keep probability
drop_block.set_keep_prob(0.1)
b = drop_block(a, training=True)

print(b[0, :, :, 0])
