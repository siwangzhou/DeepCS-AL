import tensorflow as tf
class_num = 30

def BN(input, is_training):
    return tf.layers.batch_normalization(input, training=is_training, trainable=False)

def weight_variable_vgg(name, shape, trainable=False):
    initial = tf.keras.initializers.he_normal()
    var = tf.get_variable(name=name, shape=shape, initializer=initial, trainable=trainable)
    return var

def bias_variable_vgg(shape, trainable=False):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, trainable=trainable)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(input, k_size, stride, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)

def vgg_net(data, keep_prob, train_flag):

    with tf.variable_scope('VGG'):
        w1_1 = weight_variable_vgg('w1_1', [3, 3, 1, 64])
        b1_1 = bias_variable_vgg([64])
        bn1_1 = BN(conv2d(data, w1_1) + b1_1, train_flag)
        conv1_1 = tf.nn.relu(bn1_1)

        w1_2 = weight_variable_vgg('w1_2', [3, 3, 64, 64])
        b1_2 = bias_variable_vgg([64])
        bn1_2 = BN(conv2d(conv1_1, w1_2) + b1_2, train_flag)
        conv1_2 = tf.nn.relu(bn1_2)
        pool1 = max_pool(conv1_2, 2, 2, "block1_pool")

        w2_1 = weight_variable_vgg('w2_1', [3, 3, 64, 128])
        b2_1 = bias_variable_vgg([128])
        bn2_1 = BN(conv2d(pool1, w2_1) + b2_1, train_flag)
        conv2_1 = tf.nn.relu(bn2_1)

        w2_2 = weight_variable_vgg('w2_2', [3, 3, 128, 128])
        b2_2 = bias_variable_vgg([128])
        bn2_2 = BN(conv2d(conv2_1, w2_2) + b2_2, train_flag)
        conv2_2 = tf.nn.relu(bn2_2)
        pool2 = max_pool(conv2_2, 2, 2, 'block2_pool')

        w3_1 = weight_variable_vgg('w3_1', [3, 3, 128, 256])
        b3_1 = bias_variable_vgg([256])
        bn3_1 = BN(conv2d(pool2, w3_1) + b3_1, train_flag)
        conv3_1 = tf.nn.relu(bn3_1)

        w3_2 = weight_variable_vgg('w3_2', [3, 3, 256, 256])
        b3_2 = bias_variable_vgg([256])
        bn3_2 = BN(conv2d(conv3_1, w3_2) + b3_2, train_flag)
        conv3_2 = tf.nn.relu(bn3_2)

        w3_3 = weight_variable_vgg('w3_3', [3, 3, 256, 256])
        b3_3 = bias_variable_vgg([256])
        bn3_3 = BN(conv2d(conv3_2, w3_3) + b3_3, train_flag)
        conv3_3 = tf.nn.relu(bn3_3)
        pool3 = max_pool(conv3_3, 2, 2, 'block3_pool')

        w4_1 = weight_variable_vgg('w4_1', [3, 3, 256, 512])
        b4_1 = bias_variable_vgg([512])
        bn4_1 = BN(conv2d(pool3, w4_1) + b4_1, train_flag)
        conv4_1 = tf.nn.relu(bn4_1)

        w4_2 = weight_variable_vgg('w4_2', [3, 3, 512, 512])
        b4_2 = bias_variable_vgg([512])
        bn4_2 = BN(conv2d(conv4_1, w4_2) + b4_2, train_flag)
        conv4_2 = tf.nn.relu(bn4_2)

        w4_3 = weight_variable_vgg('w4_3', [3, 3, 512, 512])
        b4_3 = bias_variable_vgg([512])
        bn4_3 = BN(conv2d(conv4_2, w4_3) + b4_3, train_flag)
        conv4_3 = tf.nn.relu(bn4_3)
        pool4 = max_pool(conv4_3, 2, 2, "block4_pool")

        w5_1 = weight_variable_vgg('w5_1', [3, 3, 512, 512])
        b5_1 = bias_variable_vgg([512])
        bn5_1 = BN(conv2d(pool4, w5_1) + b5_1, train_flag)
        conv5_1 = tf.nn.relu(bn5_1)

        w5_2 = weight_variable_vgg('w5_2', [3, 3, 512, 512])
        b5_2 = bias_variable_vgg([512])
        bn5_2 = BN(conv2d(conv5_1, w5_2) + b5_2, train_flag)
        conv5_2 = tf.nn.relu(bn5_2)

        w5_3 = weight_variable_vgg('w5_3', [3, 3, 512, 512])
        b5_3 = bias_variable_vgg([512])
        bn5_3 = BN(conv2d(conv5_2, w5_3) + b5_3, train_flag)
        conv5_3 = tf.nn.relu(bn5_3)
        pool5 = max_pool(conv5_3, 2, 2, "block5_pool")  #修改1

        pool5_flat = tf.keras.layers.Flatten()(pool5)

        w_fc1 = weight_variable_vgg('w_fc1', [pool5_flat.get_shape()[1].value, 4096]) #512
        b_fc1 = bias_variable_vgg([4096])
        fc1 = tf.nn.relu(BN(tf.matmul(pool5_flat, w_fc1) + b_fc1, train_flag))
        fc1 = tf.nn.dropout(fc1, keep_prob)

        w_fc2 = weight_variable_vgg('w_fc2', [4096, 4096])
        b_fc2 = bias_variable_vgg([4096])
        fc2 = tf.nn.relu(BN(tf.matmul(fc1, w_fc2) + b_fc2, train_flag))
        fc2 = tf.nn.dropout(fc2, keep_prob)

        w_fc3 = weight_variable_vgg('w_fc3', [4096, class_num])
        b_fc3 = bias_variable_vgg([class_num])
        fc3 = tf.matmul(fc2, w_fc3) + b_fc3

    return fc3