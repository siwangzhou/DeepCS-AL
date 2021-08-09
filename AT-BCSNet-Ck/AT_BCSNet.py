import tensorflow as tf
import numpy as np
import scipy.io as io
from skimage.metrics import structural_similarity as ssim
import vgg16

model = "test"
B = 32
sample_rate = 0.1
nb = round(B * B * sample_rate)
batch_size = 64
width, height = 96, 96
class_num = 30
dropout_rate = 0.5

W = io.loadmat("G:/DeepCS-AL/models/Kernel/%s/k.mat" % sample_rate)
W_sample = W['k']
model_dir = "./AT_BCSNet_imagenet/%s/model" % sample_rate

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

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=trainable)

def conv2d_SAME(x, W, stride=[1, 1, 1, 1]):
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

def CSNet_IT(data):
    with tf.variable_scope("CSNet_IT"):
        with tf.variable_scope("sampling"):
            X_image = img_block2col(data)
            y = Phi_X(W_sample, X_image)
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

def train_model():
    X = tf.placeholder(tf.float32, [None, width, height, 1])
    Y = tf.placeholder(tf.float32, [None, class_num])
    learning_rate_holder = tf.placeholder(tf.float32)
    training_flag_holder = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    h_concat, result_recon = CSNet_IT(X)
    result_category = vgg16.vgg_net(result_recon, keep_prob, training_flag_holder)

    with tf.variable_scope('loss'):
        loss_1 = tf.reduce_sum(tf.square(X - result_recon)) / 2 / batch_size
        loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result_category, labels=Y))
        loss = loss_1 + loss_2

    with tf.variable_scope('evaluate'):
        correct = tf.equal(tf.argmax(result_category, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    params_1 = tf.global_variables(scope="CSNet_IT")
    params_2 = tf.global_variables(scope="VGG")

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate_holder)
        train_op = optimizer.minimize(loss, var_list=params_1)

    saver_CSNet = tf.train.Saver(var_list=params_1, max_to_keep=5)
    saver_VGG = tf.train.Saver(var_list=params_2, max_to_keep=5)

    with tf.Session() as sess:
        if model == "train":
            dataset_train = generate_trainData()
            train_image = dataset_train[0][0]
            train_label = dataset_train[0][1]

            sess.run(tf.global_variables_initializer())
            saver_VGG.restore(sess, "../models/vgg_imagenet/model/Model_80.ckpt")
            saver_CSNet.restore(sess, "../models/BCSNet/%s/model/Model_99.ckpt" % sample_rate)

            for epoch in range(1, 251):

                if epoch < 200:
                    learning_rate = 1e-4
                else:
                    learning_rate = 1e-5

                np.random.seed(epoch)
                np.random.shuffle(train_image)
                np.random.seed(epoch)
                np.random.shuffle(train_label)

                loss_val = []
                loss_recon_val = []
                loss_category_val = []
                train_acc = []
                train_psnr = []
                for i in range((int)(np.shape(train_image)[0] / batch_size)):
                    batch_x = train_image[i * batch_size:(i + 1) * batch_size]
                    batch_y = train_label[i * batch_size:(i + 1) * batch_size]
                    _, train_loss, loss_recon, loss_category, train_recon, train_acc_batch = sess.run(
                        [train_op, loss, loss_1, loss_2, result_recon, accuracy],
                        feed_dict={X: batch_x, Y: batch_y,
                                   learning_rate_holder: learning_rate,
                                   training_flag_holder: False,
                                   keep_prob: 0.5})

                    loss_val.append(train_loss)
                    loss_recon_val.append(loss_recon)
                    loss_category_val.append(loss_category)
                    train_acc.append(train_acc_batch)

                    img_show = np.array(train_recon)
                    img_test = np.array(batch_x)

                    for j in range(img_test.shape[0]):
                        f_pred = img_show[j] * 255
                        f_true = img_test[j] * 255
                        train_psnr.append(psnr(f_pred, f_true))

                if epoch % 2 == 0:
                    val_im, val_cls = read_validation_batch()

                    val_loss = []
                    val_acc = []
                    pre_images = []
                    test_batch_size = 50
                    for i in range((int)(np.shape(val_im)[0] / test_batch_size)):
                        batch_x_val = val_im[i * test_batch_size:(i + 1) * test_batch_size]
                        batch_y_val = val_cls[i * test_batch_size:(i + 1) * test_batch_size]
                        val_loss_batch, val_acc_batch, predict_batch = sess.run([loss, accuracy, result_recon],
                                                                                feed_dict={X: batch_x_val,
                                                                                           Y: batch_y_val,
                                                                                           training_flag_holder: False,
                                                                                           keep_prob: 1.0})
                        if i == 0:
                            pre_images = predict_batch
                        else:
                            pre_images = np.concatenate((pre_images, predict_batch), axis=0)

                        val_loss.append(val_loss_batch)
                        val_acc.append(val_acc_batch)

                    pre_images = np.array(pre_images)

                    psnr_val = []
                    ssim_val = []
                    for j in range(pre_images.shape[0]):
                        f_pred = pre_images[j] * 255
                        f_true = val_im[j] * 255
                        f_pred = np.reshape(f_pred, (f_pred.shape[0], f_pred.shape[1]))
                        f_true = np.reshape(f_true, (f_true.shape[0], f_true.shape[1]))

                        psnr_val.append(psnr(f_pred, f_true))
                        ssim_val.append(ssim(f_pred, f_true, data_range=255))

                model_path = model_dir + "/Model_%d.ckpt" % epoch
                saver_CSNet.save(sess, model_path)


        else:  # 测试
            model_path = model_dir + "/Model_250.ckpt"
            # model_path = model_dir + "/Model_350.ckpt"
            saver_CSNet.restore(sess, model_path)
            saver_VGG.restore(sess, "../models/vgg_imagenet/model/Model_80.ckpt")
            # saver_VGG.restore(sess, "../models/vgg_cifar10/model/Model_200.ckpt")

            val_im, val_cls = read_validation_batch()

            val_loss = []
            val_acc = []
            pre_images = []
            val_batch_size = 50
            for i in range((int)(np.shape(val_im)[0] / val_batch_size)):
                batch_x_val = val_im[i * val_batch_size:(i + 1) * val_batch_size]
                batch_y_val = val_cls[i * val_batch_size:(i + 1) * val_batch_size]
                val_loss_batch, val_acc_batch, predict_batch = sess.run([loss, accuracy, result_recon],
                                                                        feed_dict={X: batch_x_val, Y: batch_y_val,
                                                                                   training_flag_holder: False,
                                                                                   keep_prob: 1.0})
                if i == 0:
                    pre_images = predict_batch
                else:
                    pre_images = np.concatenate((pre_images, predict_batch), axis=0)

                val_loss.append(val_loss_batch)
                val_acc.append(val_acc_batch)

            pre_images = np.array(pre_images)

            psnr_val = []
            ssim_val = []
            for j in range(pre_images.shape[0]):
                f_pred = pre_images[j] * 255
                f_true = val_im[j] * 255
                f_pred = np.reshape(f_pred, (f_pred.shape[0], f_pred.shape[1]))
                f_true = np.reshape(f_true, (f_true.shape[0], f_true.shape[1]))
                psnr_val.append(psnr(f_pred, f_true))
                ssim_val.append(ssim(f_pred, f_true, data_range=255))

            print("val_acc:", np.mean(val_acc), "val_psnr:", np.mean(psnr_val), "val_ssim:", np.mean(ssim_val))

def psnr(img1, img2):
    num1 = np.abs(img1 - img2)
    rmse = np.square(num1).mean()
    psnr_value = 10 * np.log10(255 ** 2 / rmse)
    return psnr_value


if __name__ == "__main__":
    train_model()