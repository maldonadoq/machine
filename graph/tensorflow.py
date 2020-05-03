import tensorflow as tf
import numpy as np

np.random.seed(0)

# Tensor dims
dim = (4,5)

# Initialize 3 Tensors
t1 = tf.placeholder(float)
t2 = tf.placeholder(float)
t3 = tf.placeholder(float)

# Computational Graph
a = t1 + t2
b = a * t3
c = tf.reduce_sum(b)

# Gradient Definition
gradt1, gradt2, gradt3 = tf.gradients(c, [t1,t2,t3])

# Session
with tf.Session() as sess:
    values = {
        t1: np.random.rand(dim[0], dim[1]),
        t2: np.random.rand(dim[0], dim[1]),
        t3: np.random.rand(dim[0], dim[1])
    }

    out = sess.run([c, gradt1, gradt2, gradt3], feed_dict=values)

    cval, gradt1val, gradt2val, gradt3val = out
    print('res: ', cval)
    print('grad 1\n', gradt1val)
    print('grad 2\n', gradt2val)
    print('grad 3\n', gradt3val)