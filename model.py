import tensorflow as tf
import tfrecords as jpeg
import numpy as np
import cv2,os,shutil,random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
np.set_printoptions(threshold=np.inf)    
def conv_layer(x_image, W_size, weight_name, stride, padding):
    W_conv1 = tf.Variable(tf.random_normal(W_size, stddev=0.1), name=weight_name)
    conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, stride, stride, 1], padding=padding) #54*54
    return conv1
    

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

mode = 'train'

bs = 50

num_training_samples = 549+550
epoch = round(num_training_samples / bs)
num_batch = epoch * 35

if mode == 'test' :
    X_Rays, label, filename_ = jpeg.read_and_decode('test.tfrecords')
    x, y_, filename = tf.train.batch([X_Rays, label, filename_],batch_size=1, capacity=16, num_threads=4)
    
if mode == 'validation':
    X_Rays, label, filename_ = jpeg.read_and_decode('validation.tfrecords')
    x, y_, filename = tf.train.batch([X_Rays, label, filename_],batch_size=1, capacity=16, num_threads=4)

if mode == 'train':
    X_Rays, label, filename_ = jpeg.read_and_decode('train.tfrecords')
    x, y_, filename = tf.train.shuffle_batch([X_Rays, label, filename_],batch_size=bs, capacity=1024, num_threads=16, min_after_dequeue=512)


y = tf.one_hot(y_, 2)

h_conv1 = conv_layer(x, [5, 5, 1, 16], 'W_conv1', 1, 'SAME')
r1 = tf.nn.relu(h_conv1)

h_conv2 = conv_layer(r1, [3, 3, 16, 16], 'W_conv2', 1, 'SAME')
r2 = tf.nn.relu(h_conv2)
pool2 = tf.nn.max_pool(r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv3 = conv_layer(pool2, [3, 3, 16, 16], 'W_conv3', 1, 'SAME')
r3 = tf.nn.relu(h_conv3)
pool3 = tf.nn.max_pool(r3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
p1 = tf.nn.max_pool(r1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
concat1 = tf.concat([p1, pool3], axis = 3)

c1 = conv_layer(concat1, [3, 3, 32, 16], 'W_c1', 1, 'SAME')
rc1 = tf.nn.relu(c1)
poolc1 = tf.nn.max_pool(rc1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv4 = conv_layer(poolc1, [3, 3, 16, 16], 'W_conv4', 1, 'SAME')
r4 = tf.nn.relu(h_conv4)
pool4 = tf.nn.max_pool(r4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

p2 = tf.nn.max_pool(concat1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
concat2 = tf.concat([p2, pool4], axis = 3)

h_conv6 = conv_layer(concat2, [3, 3, 48, 16], 'W_conv6', 1, 'SAME')

flat = tf.reshape(h_conv6, [-1, 15*31*16])
W_fc = tf.Variable(tf.random_normal([15*31*16, 2], stddev=0.1, dtype=tf.float32), 'W_fc')
b_fc = tf.Variable(tf.random_normal([2], stddev=0.01, dtype=tf.float32), 'b_fc')
fc_add = tf.matmul(flat, W_fc) + b_fc
y_conv = tf.nn.softmax(fc_add)
pred = tf.argmax(y_conv,1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_add, labels=y)) 
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy, var_list=[var for var in tf.trainable_variables()]) 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver(max_to_keep=1000, var_list=[var for var in tf.trainable_variables()])
    
with tf.Session() as sess:
    if mode == 'train':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        cnt = 1
        l = []
        for i in range(num_batch):
            _, loss = sess.run([train_step, cross_entropy])
            l.append(np.mean(loss))
            if (i+1) % epoch == 0:
                saver.save(sess, 'checkpoints/%d.ckpt'%(cnt))
                print('Save ckpt %d, Loss %g'%(cnt, np.mean(l)))
                cnt += 1
                l = []
    elif mode == 'validation':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        for cnt in range(1,1+32):
            TP = 0.
            FN = 0.
            TN = 0.
            FP = 0.
            saver.restore(sess, 'checkpoints/%d.ckpt'%(cnt))
            for i in range(100):
                img_name, predict, probability, GT = sess.run([filename ,correct_prediction, y_conv, y_])
                if GT == [0]:
                    if predict == [True]:
                        TN += 1
                    else:
                        FP += 1
                elif GT == [1]:
                    if predict == [True]:
                        TP += 1
                    else:
                        FN += 1
            SPEC = TN/(TN+FP+1e-5)
            SEN = TP/(TP+FN+1e-5)
            print('\nEpoch = %d'%cnt)
            print('TP = %g FN = %g'%(TP,FN))
            print('TN = %g FP = %g'%(TN,FP))
            print('SPEC = %f'%(SPEC))
            print('SEN = %f'%(SEN))
    elif mode == 'test':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        cnt = 32
        TP = 0.
        FN = 0.
        TN = 0.
        FP = 0.
        saver.restore(sess, 'checkpoints/%d.ckpt'%(cnt))
        for i in range(120):
            img_name, predict, probability, GT = sess.run([filename ,correct_prediction, y_conv, y_])
            if GT == [0]:
                if predict == [True]:
                    TN += 1
                else:
                    FP += 1
            elif GT == [1]:
                if predict == [True]:
                    TP += 1
                else:
                    FN += 1
        SPEC = TN/(TN+FP+1e-5)
        SEN = TP/(TP+FN+1e-5)
        print('\nEpoch = %d'%cnt)
        print('TP = %g FN = %g'%(TP,FN))
        print('TN = %g FP = %g'%(TN,FP))
        print('SPEC = %f'%(SPEC))
        print('SEN = %f'%(SEN))