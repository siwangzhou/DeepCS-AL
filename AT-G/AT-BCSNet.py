import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import scipy.io as io
import pickle
from keras.utils import to_categorical
from skimage.metrics import structural_similarity as ssim
import cv2

model="test"
B = 32
sample_rate = 0.1
nb = round(B * B * sample_rate)
batch_size = 100
width, height = 32, 32
num_classes = 10

model_dir_G = "./AT_BCSNet_G_cifar10/%s/model_G" % sample_rate
model_dir_D = "./AT_BCSNet_G_cifar10/%s/model_D" % sample_rate
output_file_name = "./AT_BCSNet_G_cifar10/%s/Log_output_AT_BCSNet_G_cifar10_%s.txt" % (sample_rate, sample_rate)
mat_save_path = "./AT_BCSNet_G_cifar10/%s/predict.mat" % sample_rate

W = io.loadmat("../DeepCS-AL/models/Kernel/%s/k.mat" % sample_rate)
W_sample = W['k']
print("W_sample:", W_sample.shape)  # (102, 1024)

def generate_trainData():
    trainDataset_path = "../datasets/imagenet/trainData/train.mat"
    train_set = io.loadmat(trainDataset_path)
    trainData = train_set['data']
    trainLabel = train_set['labels']
    return [(trainData, trainLabel)]

def read_validation_batch():
    testDataset_path = "../datasets/imagenet/testData/test.mat"
    test_set = io.loadmat(testDataset_path)
    testData = test_set['data']
    testLabel = test_set['labels']
    return testData, testLabel

def weight_variable(shape, trainable=True):
    if len(shape) == 4:
        N = shape[0] * shape[1] * (shape[2] + shape[3]) / 2
    else:
        N = shape[0] / 2
    initial = tf.random_normal(shape, stddev=np.sqrt(2.0 / N))
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable = True):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=trainable)

def conv2d_SAME(x, W, stride = [1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

def conv2d_VALID(x, W, stride):
    return tf.nn.conv2d(x, W, strides=stride, padding='VALID')

def img_block2col(image):
    img2col = []
    for i in range(width // B):
        row_ = []
        for j in range(height // B):
            block_tmp = image[:, i * B:(i + 1) * B, j * B:(j + 1) * B, :]
            block_reshape = tf.reshape(block_tmp, [-1, B * B, 1])
            if j == 0:
                row_ = block_reshape
            else:
                row_ = tf.concat([row_, block_reshape], 2)

        if i == 0:
            img2col = row_
        else:
            img2col = tf.concat([img2col, row_], 2)

    return img2col

def img_col2block(image):
    img2block = []
    for i in range(width // B):
        row_ = []
        for j in range(height // B):
            col_tmp = image[:, :, i * (height // B) + (j + 1) - 1]
            col_reshape = tf.reshape(col_tmp, [-1, B, B])

            if j == 0:
                row_ = col_reshape
            else:
                row_ = tf.concat([row_, col_reshape], 2)

        if i == 0:
            img2block = row_
        else:
            img2block = tf.concat([img2block, row_], 1)

    img2block = tf.reshape(img2block, [-1, img2block.shape[1], img2block.shape[2], 1])
    return img2block

def Phi_X(_phi, data):
    phi_batch = []
    for i in range(data.shape[2]):
        data_tmp = data[:, :, i]
        data_tmp = tf.transpose(data_tmp, [1, 0])
        data_phi_x = tf.matmul(_phi, data_tmp)
        phi_batch.append(data_phi_x)

    phi_batch = tf.transpose(phi_batch, [2, 1, 0])
    return phi_batch

def IT(y, h_conv):

    h_conv = img_block2col(h_conv)
    phi_x0 = Phi_X(W_sample, h_conv)

    Phi_inv = np.linalg.pinv(W_sample)


    h_conv_IT = h_conv + Phi_X(Phi_inv, y - phi_x0)
    h_conv_IT = img_col2block(h_conv_IT)

    return h_conv_IT

def deep_conv_block(y, data):
    c = 64
    with tf.variable_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, c])
        b_conv1 = bias_variable([c])
        h_conv1 = tf.nn.relu(conv2d_SAME(data, W_conv1) + b_conv1)

    with tf.variable_scope('conv2'):
        W_conv2 = weight_variable([3, 3, c, c])
        b_conv2 = bias_variable([c])
        h_conv2 = tf.nn.relu(conv2d_SAME(h_conv1, W_conv2) + b_conv2)

    with tf.variable_scope('conv3'):
        W_conv3 = weight_variable([3, 3, c, c])
        b_conv3 = bias_variable([c])
        h_conv3 = tf.nn.relu(conv2d_SAME(h_conv2, W_conv3) + b_conv3)

    with tf.variable_scope('conv4'):
        W_conv4 = weight_variable([3, 3, c, c])
        b_conv4 = bias_variable([c])
        h_conv4 = tf.nn.relu(conv2d_SAME(h_conv3, W_conv4) + b_conv4)

    with tf.variable_scope('conv5'):
        W_conv5 = weight_variable([3, 3, c, 1])
        b_conv5 = bias_variable([1])
        h_conv5 = conv2d_SAME(h_conv4, W_conv5) + b_conv5

    with tf.variable_scope('IT'):
        h_conv_IT = IT(y, h_conv5)

    return h_conv_IT

def deep_Net(y, data, n):
    h_conv = data
    for i in range(n):
        with tf.variable_scope("deep_conv_%d" % i, reuse=tf.AUTO_REUSE):
            h_conv = deep_conv_block(y, h_conv)
    return h_conv

def Generator(data):
    with tf.variable_scope("CSNet_IT"):

        with tf.variable_scope("sampling"):

            X_image = img_block2col(data)
            y = Phi_X(W_sample, X_image)
            print("y:", y.shape)

            y_T = tf.transpose(y, [0, 2, 1])
            y_conv = tf.reshape(y_T, [-1, width // B, height // B, y_T.shape[2]])

        with tf.variable_scope("initial_reconstruct"):
            W_conv2 = weight_variable([1, 1, nb, B * B])
            h_conv2 = conv2d_VALID(y_conv, W_conv2, stride=[1, 1, 1, 1])

        with tf.variable_scope("concat"):
            col_ = []
            for row in range(width // B):
                row_ = []
                for col in range(height // B):
                    tmp_block = h_conv2[:, row, col, :]
                    reshape_block = tf.reshape(tmp_block, [-1, B, B])
                    if col == 0:
                        row_ = reshape_block
                    else:
                        row_ = tf.concat([row_, reshape_block], 2)

                if row == 0:
                    col_ = row_
                else:
                    col_ = tf.concat([col_, row_], 1)
            h_concat = tf.reshape(col_, [-1, width, height, 1])

        with tf.variable_scope("CSNet_IT_20"):
            h_conv20_IT = deep_Net(y, h_concat, 4)

    return h_concat, h_conv20_IT

def Discriminator(X, dropout_rate, is_training=True, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        conv1 = tf.layers.conv2d(X, 64, kernel_size=[3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu, name='conv1')  #输入层无BN
        dropout1 = tf.nn.dropout(conv1, dropout_rate)

        conv2 = tf.layers.conv2d(dropout1, 128, kernel_size=[3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu, name='conv2')
        batch2 = tf.layers.batch_normalization(conv2, training=is_training)
        dropout2 = tf.nn.dropout(batch2, dropout_rate)

        conv3 = tf.layers.conv2d(dropout2, 256, kernel_size=[3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu, name='conv3')  #strides=[4, 4]
        batch3 = tf.layers.batch_normalization(conv3, training=is_training)
        dropout3 = tf.nn.dropout(batch3, dropout_rate)

        conv4 = tf.layers.conv2d(dropout3, 512, kernel_size=[3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu, name='conv4')  # ?*2*2*1024

        flatten = tf.reduce_mean(conv4, axis=[1, 2])
        logits_D_S = tf.layers.dense(flatten, 1)
        logits_D_C = tf.layers.dense(flatten, num_classes)

        out_C = tf.nn.softmax(logits_D_C)

    return logits_D_S, logits_D_C, out_C

def vgg_content(data, keep_prob):
    with tf.variable_scope('VGG', reuse=tf.AUTO_REUSE):
        conv1_1 = tf.layers.conv2d(data, 64, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv1_1 = tf.layers.batch_normalization(conv1_1, training=False)
        conv1_1 = tf.nn.relu(conv1_1)

        conv1_2 = tf.layers.conv2d(conv1_1, 64, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv1_2 = tf.layers.batch_normalization(conv1_2, training=False)
        conv1_2 = tf.nn.relu(conv1_2)
        pool1_2 = tf.layers.max_pooling2d(conv1_2, pool_size=[2, 2], strides=[2, 2], padding='same')

        conv2_1 = tf.layers.conv2d(pool1_2, 128, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv2_1 = tf.layers.batch_normalization(conv2_1, training=False)
        conv2_1 = tf.nn.relu(conv2_1)

        conv2_2 = tf.layers.conv2d(conv2_1, 128, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv2_2 = tf.layers.batch_normalization(conv2_2, training=False)
        conv2_2 = tf.nn.relu(conv2_2)
        pool2_2 = tf.layers.max_pooling2d(conv2_2, pool_size=[2, 2], strides=[2, 2], padding='same')

        conv3_1 = tf.layers.conv2d(pool2_2, 256, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv3_1 = tf.layers.batch_normalization(conv3_1, training=False)
        conv3_1 = tf.nn.relu(conv3_1)

        conv3_2 = tf.layers.conv2d(conv3_1, 256, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv3_2 = tf.layers.batch_normalization(conv3_2, training=False)
        conv3_2 = tf.nn.relu(conv3_2)

        conv3_3 = tf.layers.conv2d(conv3_2, 256, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv3_3 = tf.layers.batch_normalization(conv3_3, training=False)
        conv3_3 = tf.nn.relu(conv3_3)
        pool3_3 = tf.layers.max_pooling2d(conv3_3, pool_size=[2, 2], strides=[2, 2], padding='same')

        conv4_1 = tf.layers.conv2d(pool3_3, 512, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv4_1 = tf.layers.batch_normalization(conv4_1, training=False)
        conv4_1 = tf.nn.relu(conv4_1)

        conv4_2 = tf.layers.conv2d(conv4_1, 512, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv4_2 = tf.layers.batch_normalization(conv4_2, training=False)
        conv4_2 = tf.nn.relu(conv4_2)

        conv4_3 = tf.layers.conv2d(conv4_2, 512, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv4_3 = tf.layers.batch_normalization(conv4_3, training=False)
        conv4_3 = tf.nn.relu(conv4_3)
        pool4_3 = tf.layers.max_pooling2d(conv4_3, pool_size=[2, 2], strides=[2, 2], padding='same')

        conv5_1 = tf.layers.conv2d(pool4_3, 512, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv5_1 = tf.layers.batch_normalization(conv5_1, training=False)
        conv5_1 = tf.nn.relu(conv5_1)

        conv5_2 = tf.layers.conv2d(conv5_1, 512, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv5_2 = tf.layers.batch_normalization(conv5_2, training=False)
        conv5_2 = tf.nn.relu(conv5_2)

        conv5_3 = tf.layers.conv2d(conv5_2, 512, kernel_size=[3, 3], kernel_initializer='he_normal', trainable=False, strides=[1, 1], padding='same')
        conv5_3 = tf.layers.batch_normalization(conv5_3, training=False)
        conv5_3 = tf.nn.relu(conv5_3)
    return conv3_3, conv5_3

def build_GAN(x_real, dropout_rate, is_training):
    h_concat, fake_images = Generator(x_real)

    x_content = vgg_content(x_real, dropout_rate)
    recon_content = vgg_content(fake_images, dropout_rate)

    D_real_logits_S, D_real_logits_C, D_real_prob = Discriminator(x_real, dropout_rate, is_training)
    D_fake_logits_S, D_fake_logits_C, D_fake_prob = Discriminator(fake_images, dropout_rate, is_training, reuse=True)

    return D_real_logits_S, D_real_logits_C, D_real_prob, D_fake_logits_S, D_fake_logits_C, D_fake_prob, fake_images, x_content, recon_content


def loss_accuracy(D_real_logits_S, D_real_logits_C, D_real_prob,
                  D_fake_logits_S, D_fake_logits_C, D_fake_prob, label, X, X_fake):
    D_S_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits_S, labels=tf.ones_like(D_real_logits_S, dtype=tf.float32)))
    D_S_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_S, labels=tf.zeros_like(D_fake_logits_S, dtype=tf.float32)))

    D_C_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_real_logits_C, labels=label))
    D_C_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_fake_logits_C, labels=label))
    D_L = 0.5*D_S_real + 0.5*D_S_fake + D_C_real

    G_M = tf.reduce_mean(tf.square(X - X_fake)) / 2.0
    G_S_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_S, labels=tf.ones_like(D_fake_logits_S, dtype=tf.float32)))
    G_C_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_fake_logits_C, labels=label))
    G_L = G_M + 0.0001*G_S_fake + 0.0001*G_C_fake

    prediction_fake = tf.equal(tf.argmax(D_fake_prob, 1), tf.argmax(label, 1))
    accuracy_fake = tf.reduce_mean(tf.cast(prediction_fake, tf.float32))

    prediction_real = tf.equal(tf.argmax(D_real_prob, 1), tf.argmax(label, 1))
    accuracy_real = tf.reduce_mean(tf.cast(prediction_real, tf.float32))

    return D_L, G_L, accuracy_fake, accuracy_real, D_S_real, D_S_fake, D_C_real, D_C_fake, G_C_fake


def optimizer(D_Loss, G_Loss, lr_g, lr_d):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        all_vars = tf.global_variables()
        D_vars = [var for var in all_vars if var.name.startswith('Discriminator')]
        G_vars = [var for var in all_vars if var.name.startswith('CSNet_IT')]

        d_train_opt = tf.train.AdamOptimizer(lr_d, beta1=0.5, name='d_optimiser').minimize(D_Loss, var_list=D_vars)
        g_train_opt = tf.train.AdamOptimizer(lr_g, beta1=0.9, beta2=0.999, name='g_optimiser').minimize(G_Loss, var_list=G_vars)  #beta1

    return d_train_opt, g_train_opt

def psnr(img1, img2):
    num1 = np.abs(img1-img2)
    rmse = np.square(num1).mean()
    psnr_value = 10*np.log10(255**2/rmse)
    return psnr_value

def Perceptual_Similarity(x_content, recon_content):
    ps = tf.reduce_mean(tf.square(x_content - recon_content)) / tf.reduce_mean(tf.square(x_content))
    return ps

def train_model():
    tf.reset_default_graph()

    if model == 'train':
        dataset_train = generate_trainData()
        train_image = dataset_train[0][0]
        train_label = dataset_train[0][1]

        dataset_test = generate_testData()
        test_image = dataset_test[0][0]
        test_label = dataset_test[0][1]
    else:
        dataset_test = generate_testData()
        test_image = dataset_test[0][0]
        test_label = dataset_test[0][1]

    X = tf.placeholder(tf.float32, [None, width, height, 1])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    learning_rate_holder = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    lr_rate = 2e-4
    GAN_model = build_GAN(X, keep_prob, is_training)
    D_real_logits_S, D_real_logits_C, D_real_prob, D_fake_logits_S, D_fake_logits_C, D_fake_prob, fake_data, x_content, recon_content = GAN_model


    loss_acc = loss_accuracy(D_real_logits_S, D_real_logits_C, D_real_prob,
                             D_fake_logits_S, D_fake_logits_C, D_fake_prob, Y, X, fake_data)
    D_L, G_L, accuracy_fake, accuracy_real, D_S_real, D_S_fake, D_C_real, D_C_fake, G_C_fake = loss_acc

    x_out, x_output = vgg_content(X, keep_prob)
    recon_out, recon_output = vgg_content(fake_data, keep_prob)
    ps_val = Perceptual_Similarity(x_out, recon_out)

    D_optimizer, G_optimizer = optimizer(D_L, G_L, learning_rate_holder, lr_rate)

    all_vars = tf.global_variables()
    G_vars = tf.trainable_variables(scope="CSNet_IT")
    D_vars = [var for var in all_vars if var.name.startswith('Discriminator')]
    VGG_vars = [var for var in all_vars if var.name.startswith('VGG')]

    saver_G = tf.train.Saver(var_list=G_vars, max_to_keep=8)
    saver_D = tf.train.Saver(var_list=D_vars, max_to_keep=8)
    saver_VGG = tf.train.Saver(var_list=VGG_vars, max_to_keep=5)

    with tf.Session() as sess:
        if model == "train":
            sess.run(tf.global_variables_initializer())
            saver_G.restore(sess, "E:/xiaoning/20201119_ER_CSNet/CSNet+IT_96/%s/model/Model_99.ckpt" % sample_rate)

            for epoch in range(1, 101):

                if epoch < 50:
                    learning_rate = 1e-3
                elif epoch < 80:
                    learning_rate = 1e-4
                else:
                    learning_rate = 1e-5

                np.random.seed(epoch)
                np.random.shuffle(train_image)
                np.random.seed(epoch)
                np.random.shuffle(train_label)

                train_accuracies_fake, train_accuracies_real, train_D_losses, train_G_losses, train_psnr_val = [], [], [], [], []
                for i in range((int)(np.shape(train_image)[0] / batch_size)):
                    batch_x = train_image[i * batch_size:(i + 1) * batch_size]
                    batch_y = train_label[i * batch_size:(i + 1) * batch_size]
                    train_feed_dict = {X: batch_x, Y:batch_y, learning_rate_holder: learning_rate, keep_prob: 0.5, is_training: True}

                    D_optimizer.run(feed_dict=train_feed_dict)
                    G_optimizer.run(feed_dict=train_feed_dict)

                    train_pre, train_D_loss, train_G_loss, train_accuracy_fake, train_accuracy_real = \
                        sess.run([fake_data, D_L, G_L, accuracy_fake, accuracy_real], feed_dict=train_feed_dict)

                    train_D_losses.append(train_D_loss)
                    train_G_losses.append(train_G_loss)
                    train_accuracies_fake.append(train_accuracy_fake)
                    train_accuracies_real.append(train_accuracy_real)

                    img_show = np.array(train_pre)
                    img_test = np.array(batch_x)

                    for j in range(img_test.shape[0]):
                        f_pred = img_show[j] * 255
                        f_true = img_test[j] * 255
                        train_psnr_val.append(psnr(f_pred, f_true))

                if epoch % 2 == 0:
                    test_accuracies_fake, test_accuracies_real, test_psnr_val = [], [], []
                    D_S_real_res, D_S_fake_res, D_C_real_res, D_C_fake_res, G_C_fake_res = [], [], [], [], []
                    for i in range((int)(np.shape(test_image)[0] / batch_size)):
                        test_batch_x = test_image[i * batch_size : (i+1) * batch_size]
                        test_batch_y = test_label[i * batch_size : (i+1) * batch_size]
                        test_feed_dict = {X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, is_training: False}

                        pred_image, test_D_loss, test_G_loss, test_accuracy_fake, test_accuracy_real=\
                            sess.run([fake_data, D_L, G_L, accuracy_fake, accuracy_real], feed_dict=test_feed_dict)

                        test_accuracies_fake.append(test_accuracy_fake)
                        test_accuracies_real.append(test_accuracy_real)

                        img_show = np.array(pred_image)
                        img_test = np.array(test_batch_x)

                        for j in range(img_test.shape[0]):
                            f_pred = img_show[j] * 255
                            f_true = img_test[j] * 255
                            test_psnr_val.append(psnr(f_pred, f_true))

                model_path_G = model_dir_G + "/Model_%d.ckpt" % epoch
                model_path_D = model_dir_D + "/Model_%d.ckpt" % epoch
                saver_G.save(sess, model_path_G)
                saver_D.save(sess, model_path_D)

        else:
            model_path_G = model_dir_G + "/Model_100.ckpt"
            model_path_D = model_dir_D + "/Model_100.ckpt"

            saver_G.restore(sess, model_path_G)
            saver_D.restore(sess, model_path_D)
            saver_VGG.restore(sess, "../DeepCS-AL/models/vgg_content/model/Model_80.ckpt")

            test_accuracies_fake, test_accuracies_real, test_psnr_val, pre_images = [], [], [], []
            res, res_fake, res_logit = [], [], []
            ps, ssim_val = [], []
            test_batch_size = 100
            for i in range((int)(np.shape(val_im)[0] / test_batch_size)):
                test_batch_x = val_im[i * test_batch_size: (i + 1) * test_batch_size]
                test_batch_y = val_cls[i * test_batch_size: (i + 1) * test_batch_size]
                test_feed_dict = {X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, is_training: False}

                ps_batch = sess.run(ps_val, feed_dict=test_feed_dict)
                ps.append(ps_batch)

                pred_image = fake_data.eval(feed_dict=test_feed_dict)
                test_accuracy_fake = accuracy_fake.eval(feed_dict=test_feed_dict)
                test_accuracy_real = accuracy_real.eval(feed_dict=test_feed_dict)

                test_accuracies_fake.append(test_accuracy_fake)
                test_accuracies_real.append(test_accuracy_real)

                if i == 0:
                    pre_images = pred_image
                else:
                    pre_images = np.concatenate((pre_images, pred_image), axis=0)

                img_show = np.array(pred_image)
                img_test = np.array(test_batch_x)
                img_show = np.squeeze(img_show)
                img_test = np.squeeze(img_test)
                for j in range(img_test.shape[0]):
                    f_pred = img_show[j] * 255
                    f_true = img_test[j] * 255
                    test_psnr_val.append(psnr(f_pred, f_true))
                    ssim_val.append(ssim(f_pred, f_true, data_range=255))

            pre_images = np.array(pre_images)
            for j in range(pre_images.shape[0]):
                f_pred = pre_images[j] * 255
                f_true = val_im[j] * 255
                f_pred = np.squeeze(f_pred)
                f_true = np.squeeze(f_true)
                num = num + 1
                test_psnr_val.append(psnr(f_pred, f_true))
                ssim_val.append(ssim(f_pred, f_true, data_range=255))

            pre_images = np.array(pre_images)
            result = pre_images * 255
            # io.savemat(mat_save_path, {"data": result, "labels": val_cls})

if __name__ == "__main__":
    train_model()
