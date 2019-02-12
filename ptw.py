import numpy as np
import tensorflow as tf
import cv2 as cv
print("import complete!")

image_list = []
label_list = []
for i in range(1,901):
    name = 'C:/Users/sfsfk/Desktop/develope/tensorflow/ssuface/data/pencilCase/test'
    name = name + str(i) + '.jpg'
    image_list.append(name)
    label_list.append([1,0,0])
for i in range(1,901):
    name = 'C:/Users/sfsfk/Desktop/develope/tensorflow/ssuface/data/tissue/tissue'
    name = name + str(i) + '.jpg'
    image_list.append(name)
    label_list.append([0,1,0])
for i in range(1,901):
    name = 'C:/Users/sfsfk/Desktop/develope/tensorflow/ssuface/data/watch/watch'
    name = name + str(i) + '.jpg'
    image_list.append(name)
    label_list.append([0,0,1])

def _read_list_image(input):
    return [cv.imread(input[x], cv.IMREAD_GRAYSCALE) for x in range(2700)]

img = np.stack(_read_list_image(image_list))
img = tf.reshape(img, shape=[-1,2304])
img = tf.to_float(img,name='ToFloat')

label = np.array(label_list, dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices((img,label))
dataset = dataset.repeat()
dataset = dataset.shuffle(5000)
dataset = dataset.batch(20)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

#*******************************************************#

x = tf.placeholder("float", shape=[None, 2304])
y = tf.placeholder("float", shape=[None, 3])
x_image = tf.reshape(x, [-1,48,48,1])

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

kernel_size = 5;
chennel_1 = 32;
chennel_2 = 64;
chennel_3 = 128;

w_conv1 = weight_variable([kernel_size,kernel_size,1,chennel_1])
b_conv1 = bias_variable([chennel_1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([kernel_size, kernel_size, chennel_1, chennel_2])
b_conv2 = bias_variable([chennel_2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_conv3 = weight_variable([kernel_size, kernel_size, chennel_2, chennel_3])
b_conv3 = bias_variable([chennel_3])
h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
h_pool3 = h_conv3

h_pool2_flat = tf.reshape(h_pool3, [-1, 12*12*chennel_3])

w_fc1 = weight_variable([12 * 12 * chennel_3, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 512])
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

w_fc3 = weight_variable([512, 3])
b_fc3 = bias_variable([3])

y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)

# cross_entropy = -tf.reduce_sum(y * tf.log(y_conv)) y_conv,1e-10,1.0
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv + 1e-10))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer)

for i in  range(201):
    x_, y_ = sess.run(next_element)
    if i%10 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:x_, y:y_, keep_prob:1.0})
        print("\nstep %d, training accuracy %g"%(i, train_accuracy))
        pred_y = sess.run(y_conv, feed_dict={x:x_, keep_prob:1.0})
        #print("kernel : ",sess.run(w_conv1))
        for q in range(20):
            print("predict : ",sess.run(tf.argmax(pred_y[q])), " , y : ",sess.run(tf.argmax(y_[q])))
    sess.run(train_step, feed_dict={x:x_, y:y_, keep_prob: 0.5})
