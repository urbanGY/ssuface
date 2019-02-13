import numpy as np
import tensorflow as tf
import cv2 as cv
print("import complete!")

image_list = []
label_list = []
for i in range(901,1001): #100개
    name = 'C:/Users/sfsfk/Desktop/develope/tensorflow/ssuface/data/pencilCase/test'
    name = name + str(i) + '.jpg'
    image_list.append(name)
    label_list.append([1,0,0])
for i in range(901,925): #24개
    name = 'C:/Users/sfsfk/Desktop/develope/tensorflow/ssuface/data/tissue/tissue'
    name = name + str(i) + '.jpg'
    image_list.append(name)
    label_list.append([0,1,0])
for i in range(901,919): #18개
    name = 'C:/Users/sfsfk/Desktop/develope/tensorflow/ssuface/data/watch/watch'
    name = name + str(i) + '.jpg'
    image_list.append(name)
    label_list.append([0,0,1])

def _read_list_image(input):
    return [cv.imread(input[x], cv.IMREAD_GRAYSCALE) for x in range(142)]

img = np.stack(_read_list_image(image_list))
img = tf.reshape(img, [-1,48,48,1])
img = tf.to_float(img,name='ToFloat')

y_data = np.array(label_list, dtype=np.float32)


x = tf.placeholder(tf.float32, shape=[None, 48, 48, 1])
y = tf.placeholder(tf.float32, shape=[None, 3])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=5e-2))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

# 첫번째 Pooling layer
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 두번째 convolutional layer - 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)합니다.
W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

# 두번째 pooling layer.
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 세번째 convolutional layer
W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

# Fully Connected Layer 1 - 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 됩니다.
# 이를 1024개의 특징들로 맵핑(maping)합니다.
W_fc1 = tf.Variable(tf.truncated_normal(shape=[12 * 12 * 128, 1024], stddev=5e-2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_conv3_flat = tf.reshape(h_conv3, [-1, 12*12*128])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

# Dropout - 모델의 복잡도를 컨트롤합니다. 특징들의 co-adaptation을 방지합니다.
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 3], stddev=5e-2))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[3]))
logits = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
y_pred = tf.nn.softmax(logits)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    x_data = sess.run(img)
    saver.restore(sess, './model/model.ckpt')
    test_accuracy = sess.run(accuracy, feed_dict = {x:x_data, y:y_data, keep_prob:1.0})
    print("test accuracy : %f" % (test_accuracy))
    for i in range(142):
        prediction = sess.run(y_pred, feed_dict={x:x_data, keep_prob:1.0})
        print("%d , prediction : %f , label : %f" % (i,sess.run(tf.argmax(prediction[i])), sess.run(tf.argmax(y_data[i]))))
