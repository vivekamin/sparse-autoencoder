import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#mnist = tf.contrib.learn.datasets.load_dataset("mnist")


def kl_divergence(p, p_hat):
    return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)

def mosaicW1(images, img_h_w, num_of_imgs, f_name):

    figure, axes = plt.subplots(nrows=num_of_imgs, ncols=num_of_imgs)
    # if file_name == "weights":
    #     figure, axes = plt.subplots(nrows=10, ncols=20)

    index = 0
    for axis in axes.flat:
        image = axis.imshow(images[index, :].reshape(img_h_w, img_h_w),
                            cmap=plt.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    file=f_name+".png"
    plt.title(f_name, y=12.00,x=-7.0)
    plt.savefig(file)
    print("plotted ", file)
    plt.close()


P_LIST = [0.01, 0.1, 0.5, 0.8]

for p_ in P_LIST:
    learning_rate = 1e-3
    epochs = 40
    batch_size = 100
    reg_term_lambda = 1e-3
    p = p_
    beta = 3
    std1 = math.sqrt(6) / math.sqrt(784+200+1)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 784])

    W1 = tf.Variable(tf.random_normal([784, 200], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([200]), name='b1')

    W2 = tf.Variable(tf.random_normal([200, 784], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([784]), name='b2')

    linear_layer_one_output = tf.add(tf.matmul(x, W1), b1)
    layer_one_output = tf.nn.sigmoid(linear_layer_one_output)

    linear_layer_two_output = tf.add(tf.matmul(layer_one_output,W2),b2)
    y_ = tf.nn.sigmoid(linear_layer_two_output)

    diff = y_ - x

    p_hat = tf.reduce_mean(tf.clip_by_value(layer_one_output,1e-10,1.0),axis=0)

    #p_hat = tf.reduce_mean(layer_one_output,axis=1)
    kl = kl_divergence(p, p_hat)

    cost= tf.reduce_mean(tf.reduce_sum(diff**2,axis=1)) + reg_term_lambda*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) + beta*tf.reduce_sum(kl)

    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate ,beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

    init_op = tf.global_variables_initializer()

    print("Running for P = ", p)
    with tf.Session() as sess:
       # initialise the variables
       sess.run(init_op)
       total_batch = int(len(mnist.train.labels) / batch_size)
       for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                #print (batch_y)
                _, c = sess.run([optimiser, cost],
                             feed_dict={x: batch_x})

                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
            if((epoch + 1) % epochs == 0):
               images = W1.eval(sess)
               images = images.transpose()
               mosaicW1(images, 28, 10, f_name="weights For P="+str(p))
               # output_images = y_.eval(feed_dict={x: batch_x}, session=sess)
               # visualizeW1(output_images, 28, 10, epoch, file_name = "output ")
            if ((epoch + 1) % epochs == 0):
                output_images = y_.eval(feed_dict={x: batch_x}, session=sess)
                mosaicW1(output_images, 28, 10, f_name="output For P="+str(p))
                input_image = batch_x
                mosaicW1(input_image, 28, 10, f_name="input For P="+str(p))