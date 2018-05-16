# -*- coding: utf-8 -*-

# Sample code to use string producer.
"""
import tensorflow as tf

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

filenames0 = tf.train.match_filenames_once("train/0/*.jpg")
filenames1 = tf.train.match_filenames_once("train/1/*.jpg")
#filenames2 = tf.train.match_filenames_once("data3/cosas/2/*.jpg")

filename_queue0 = tf.train.string_input_producer(filenames0, shuffle=False)
filename_queue1 = tf.train.string_input_producer(filenames1, shuffle=False)
#filename_queue2 = tf.train.string_input_producer(filenames2, shuffle=False)

reader0 = tf.WholeFileReader()
reader1 = tf.WholeFileReader()
#reader2 = tf.WholeFileReader()

key0, file_image0 = reader0.read(filename_queue0)
key1, file_image1 = reader1.read(filename_queue1)
#key2, file_image2 = reader2.read(filename_queue2)

image0, label0 = tf.image.decode_jpeg(file_image0), [0.]  # key0
image0 = tf.image.rgb_to_grayscale(image0, name=None)
image0 = tf.image.resize_image_with_crop_or_pad(image0, 80, 140)
image0 = tf.reshape(image0, [80, 140, 1])

image1, label1 = tf.image.decode_jpeg(file_image1), [1.]  # key1
image1 = tf.image.rgb_to_grayscale(image1, name=None)
image1 = tf.image.resize_image_with_crop_or_pad(image1, 80, 140)
image1 = tf.reshape(image1, [80, 140, 1])

# image2, label2 = tf.image.decode_jpeg(file_image2), [2.]  # key2
# image2 = tf.image.rgb_to_grayscale(image2, name=None)
# image2 = tf.image.resize_image_with_crop_or_pad(image2, 80, 140)
# image2 = tf.reshape(image1, [80, 140, 1])

image0 = tf.to_float(image0) / 256. - 0.5
image1 = tf.to_float(image1) / 256. - 0.5
#image2 = tf.to_float(image2) / 256. - 0.5

batch_size = 4
min_after_dequeue = 10  # 10000
capacity = min_after_dequeue + 3 * batch_size

example_batch0, label_batch0 = tf.train.shuffle_batch([image0, label0], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch1, label_batch1 = tf.train.shuffle_batch([image1, label1], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

#example_batch2, label_batch2 = tf.train.shuffle_batch([image2, label2], batch_size=batch_size, capacity=capacity,
 #                                                     min_after_dequeue=min_after_dequeue)

example_batch = tf.concat(values=[example_batch0, example_batch1], axis=0)
label_batch = tf.concat(values=[label_batch0, label_batch1], axis=0)

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

o1 = tf.layers.conv2d(inputs=example_batch, filters=32, kernel_size=3, activation=tf.nn.relu)
o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 2, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
y = tf.layers.dense(inputs=h, units=1, activation=tf.nn.sigmoid)

cost = tf.reduce_sum(tf.square(y - label_batch))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())


    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(200):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(y))
            print(sess.run(label_batch))
            print("Error:", sess.run(cost))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
"""


# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np

def one_hot(x, n):
    o_h = np.zeros(n)
    o_h[x] = 1.
    return o_h


num_classes = 3
batch_size = 10


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(i, num_classes)  # [one_hot(float(i), num_classes)]
        #image = tf.image.rgb_to_grayscale(image, name=None)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 3])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=5, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        o5 = tf.layers.conv2d(inputs=o4, filters=128, kernel_size=3, activation=tf.nn.relu)
        o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)

        h1 = tf.layers.flatten(o6)
        h1 = tf.layers.dense(inputs=h1, units=128, activation=tf.nn.relu)
        #h1_dropout = tf.layers.dropout(h1, rate=0.5, training=is_training)
        h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu)
        #h2_dropout = tf.layers.dropout(h2, rate=0.5, training=is_training)

        y = tf.layers.dense(inputs=h2, units=num_classes, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["train/apu_nahasapeemapetilon/*.jpg", "train/homer_simpson/*.jpg", "train/krusty_the_clown/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["valid/apu_nahasapeemapetilon/*.jpg", "valid/homer_simpson/*.jpg", "valid/krusty_the_clown/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["test/apu_nahasapeemapetilon/*.jpg", "test/homer_simpson/*.jpg", "test/krusty_the_clown/*.jpg"], batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, tf.float32)))
#cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test, tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.
f = open ("Error.txt", "w")
saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(500):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))
            error = sess.run(cost_valid)
            error_test = sess.run(cost)
            print("Error de validacion:", error)
            print("Error de entrenamiento:", error_test)
            errorstr = str(error).replace(".", ",")
            f.write(errorstr + "\n")

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    mal = 0
    result = []
    label = []
    muestras = 0
    for _ in range(15):
        re = sess.run(example_batch_test_predicted)
        la = sess.run(label_batch_test)
        result.extend(re)
        label.extend(la)

    for b, r in zip(label, result):
        muestras += 1
        if tf.argmax(b) != tf.argmax(r):
            mal = mal + 1
            print b, "-->", r, "Mal clasificado"
        else:
            print b, "-->", r
        print "----------------------------------------------------------------------------------"
    print "Total de muestras: ", muestras
    print "Clasifico mal: ", mal

    coord.request_stop()
    coord.join(threads)