import tensorflow as tf
from nets.dropblock import DropBlock2D

from scripts.utils import write_csv
import timeit

start_time = timeit.default_timer()
skipped_time = 0

# only support `channels_last` data format
a = tf.ones([2, 10, 10, 3])

drop_block = DropBlock2D(keep_prob=0.8, block_size=3)

b = drop_block(a, training=True)

print_time = timeit.default_timer()
print(b[0, :, :, 0])
skipped_time += timeit.default_timer() - print_time

# update keep probability
drop_block.set_keep_prob(0.1)
b = drop_block(a, training=True)

print_time = timeit.default_timer()
print(b[0, :, :, 0])
skipped_time += timeit.default_timer() - print_time

time = timeit.default_timer() - start_time - skipped_time

write_csv(__file__, None, None, None, time)
