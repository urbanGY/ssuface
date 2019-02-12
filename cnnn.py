import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x): # compression 2x2 -> 1x1
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
h_pool3 = h_conv3

w_fc1 = weight_variable([7 * 7 * 128, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool3, [-1, 7*7*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
cross_entropy = -tf.reduce_sum(y*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in  range(501):
    batch = mnist.train.next_batch(50)
    if i%10 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
        print("\nstep %d, training accuracy %g"%(i, train_accuracy))
        pred_y = sess.run(y_conv, feed_dict={x:batch[0], keep_prob:1.0})
        for q in range(15):
            print("predict : ",sess.run(tf.argmax(pred_y[q])), " , y : ",sess.run(tf.argmax(batch[1][q])))
    sess.run(train_step, feed_dict={x:batch[0], y:batch[1], keep_prob: 0.5})
