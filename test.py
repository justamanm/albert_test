import tensorflow as tf

# ret = tf.cast(tf.greater_equal([[0.0], [0.4], [0.5], [0.6], [1.2], [-0.4]], 0.0), dtype=tf.int32)
# out1 = tf.reshape([[0.0], [0.4], [0.5]], [-1])
# out2 = tf.reshape([[0.0], [0.8], [0.5]], [-1])
# losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=out1, labels=tf.cast([0, 1, 1], dtype=tf.float32))
# loss1 = tf.reduce_mean(losses1)
# losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=out2, labels=tf.cast([0, 1, 1], dtype=tf.float32))
# loss2 = tf.reduce_mean(losses2)

# output = tf.Variable([[0.5, 0.5], [0.6,0.4], [0.4,0.6], [0.2,0.8], [0.7,0.3]])
# output = tf.Variable([[0.5, 0.5], [0.8,0.2], [0.4,0.6], [0.2,0.8], [0.7,0.3]])
# one_hot_labels = tf.one_hot([0, 1, 1, 0, 1], depth=2, dtype=tf.float32)
# per_example_loss = -tf.reduce_sum(one_hot_labels * output, axis=-1)
# ret_loss = tf.reduce_mean(per_example_loss)



# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
    # ret1 = sess.run(losses1)
    # ret2 = sess.run(losses2)
    # print(ret1, ret2)
    # one_hot_label = sess.run(one_hot_labels)
    # per_example_los = sess.run(per_example_loss)
    # ret_los = sess.run(ret_loss)
    # print(one_hot_label)
    # print(per_example_los)
    # print(ret_los)

def run():
    i = 1
    while True:
        yield i
        i += 1
        if i == 10:
            return


for i, x in enumerate(run()):
    print("-"*30)
    print(i)
    print(x)