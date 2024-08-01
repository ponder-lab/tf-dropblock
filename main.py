import tensorflow as tf
from nets.dropblock import DropBlock2D

# only support `channels_last` data format
a = tf.ones([2, 10, 10, 3])

drop_block = DropBlock2D(keep_prob=0.8, block_size=3)
b = drop_block(a, training=True)

print(b[0, :, :, 0])

# update keep probability
drop_block.set_keep_prob(0.1)
b = drop_block(a, training=True)

print(b[0, :, :, 0])
