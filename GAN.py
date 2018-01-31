#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

#from datetime import datetime
#import matplotlib.pyplot as plt


class GAN(object):
    def __init__(self, sess, epoch, batch_size, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = "GAN"     # name for checkpoint

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist': # fix
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = 62         # dimension of noise-vector
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size # 700 = 70000 / 100

        elif dataset_name == 'cifar10':
            # parameters
            self.input_height = 32
            self.input_width = 32
            self.output_height = 32
            self.output_width = 32

            self.z_dim = 100         # dimension of noise-vector
            self.c_dim = 3  # color dimension

            # train
            #self.learning_rate = 0.0002 # 1e-3, 1e-4
            self.learningRateD = 1e-3
            self.learningRateG = 1e-4
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load cifar10
            self.data_X, self.data_y = load_cifar10()

            #validatin images

            '''
            # revice image data // M*N*3 // RGB float32 : value must set between 0. with 1.
            vMin = np.amin(self.data_X[0])
            vMax = np.amax(self.data_X[0])
            img_arr = self.data_X[0].reshape(32*32*3,1) # flatten
            for i, v in enumerate(img_arr):
                img_arr[i] = (v-vMin)/(vMax-vMin)
            img_arr = img_arr.reshape(32,32,3) # M*N*3

            # matplot display
            plt.subplot(1,1,1),plt.imshow(img_arr, interpolation='nearest')
            plt.title("pred.:{}".format(np.argmax(self.data_y[0]),fontsize=10))
            plt.axis("off")

            imgName = "{}.png".format(datetime.now())
            imgName = imgName.replace(":","_")
            #plt.savefig(os.path.join(".\\pic_result",imgName))
            plt.savefig(imgName)
            plt.show()            
            '''

            # get number of batches for a single epoch
            #print(len(self.data_X),len(self.data_y))
            #self.num_batches = self.data_X.get_shape()[0] // self.batch_size
            self.num_batches = len(self.data_X) // self.batch_size
            #print(self.num_batches)
        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):

            if self.dataset_name == 'cifar10':
                print("D:",x.get_shape()) # 32, 32, 3 = 3072
                net = lrelu(conv2d(x, 64, 5, 5, 2, 2, name='d_conv1'+'_'+self.dataset_name))
                print("D:",net.get_shape())
                net = lrelu(bn(conv2d(net, 128, 5, 5, 2, 2, name='d_conv2'+'_'+self.dataset_name), is_training=is_training, scope='d_bn2'))
                print("D:",net.get_shape())
                net = lrelu(bn(conv2d(net, 256, 5, 5, 2, 2, name='d_conv3'+'_'+self.dataset_name), is_training=is_training, scope='d_bn3'))
                print("D:",net.get_shape())
                net = lrelu(bn(conv2d(net, 512, 5, 5, 2, 2, name='d_conv4'+'_'+self.dataset_name), is_training=is_training, scope='d_bn4'))
                print("D:",net.get_shape())
                net = tf.reshape(net, [self.batch_size, -1])
                print("D:",net.get_shape())
                out_logit = linear(net, 1, scope='d_fc5'+'_'+self.dataset_name)
                print("D:",net.get_shape())
                out = tf.nn.sigmoid(out_logit)
                print("D:",out.get_shape())
                print("------------------------")

            else: # mnist / fashion mnist
                #print(x.get_shape())
                net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'+'_'+self.dataset_name))
                net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'+'_'+self.dataset_name), is_training=is_training, scope='d_bn2'))
                net = tf.reshape(net, [self.batch_size, -1])
                net = lrelu(bn(linear(net, 1024, scope='d_fc3'+'_'+self.dataset_name), is_training=is_training, scope='d_bn3'))
                out_logit = linear(net, 1, scope='d_fc4'+'_'+self.dataset_name)
                out = tf.nn.sigmoid(out_logit)

            return out, out_logit, net

    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):

            if self.dataset_name == 'cifar10':
                h_size = 32
                h_size_2 = 16
                h_size_4 = 8
                h_size_8 = 4
                h_size_16 = 2

                print("G:",z.get_shape())
                net = linear(z, 512*h_size_16*h_size_16, scope='g_fc1'+'_'+self.dataset_name)
                print("G:",net.get_shape())
                net = tf.nn.relu(
                    bn(tf.reshape(net, [self.batch_size, h_size_16, h_size_16, 512]),is_training=is_training, scope='g_bn1')
                    )
                print("G:",net.get_shape())
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_8, h_size_8, 256], 5, 5, 2, 2, name='g_dc2'+'_'+self.dataset_name),is_training=is_training, scope='g_bn2')
                    )
                print("G:",net.get_shape())
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_4, h_size_4, 128], 5, 5, 2, 2, name='g_dc3'+'_'+self.dataset_name),is_training=is_training, scope='g_bn3')
                    )
                print("G:",net.get_shape())
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_2, h_size_2, 64], 5, 5, 2, 2, name='g_dc4'+'_'+self.dataset_name),is_training=is_training, scope='g_bn4')
                    )
                print("G:",net.get_shape())
                out = tf.nn.tanh(
                    deconv2d(net, [self.batch_size, self.output_height, self.output_width, self.c_dim], 5, 5, 2, 2, name='g_dc5'+'_'+self.dataset_name)
                    )
                print("G:",out.get_shape())
                print("------------------------")

            else: # mnist / fashon mnist
                h_size = 28
                h_size_2 = 14
                h_size_4 = 7

                net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'+'_'+self.dataset_name), is_training=is_training, scope='g_bn1'))
                net = tf.nn.relu(bn(linear(net, 128 * h_size_4 * h_size_4, scope='g_fc2'+'_'+self.dataset_name), is_training=is_training, scope='g_bn2'))
                net = tf.reshape(net, [self.batch_size, h_size_4, h_size_4, 128]) #  8 8  128
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_2, h_size_2, 64], 4, 4, 2, 2, name='g_dc3'+'_'+self.dataset_name), is_training=is_training,scope='g_bn3')
                    )

                out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, self.output_height, self.output_width, self.c_dim], 4, 4, 2, 2, name='g_dc4'+'_'+self.dataset_name))

            return out

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size # 100

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learningRateD, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learningRateG, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)
            #self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim)) # 100, 62
        self.test_images = self.data_X[0:self.batch_size]

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '\\' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            print(" [!] START_EPOCH is ",start_epoch," START_BATCH_ID is ", start_batch_id)
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                #self.sess.run([self.g_optim], feed_dict={self.inputs: batch_images, self.z: batch_z})
                # update G twice to make sure that d_loss does not go to zero
                _, _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_optim, self.g_sum, self.g_loss], feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.inputs: self.test_images})
                    tot_num_samples = min(self.sample_num, self.batch_size) # 64
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples))) # 8
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples))) # 8
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                '.\\' + self.result_dir + '\\' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size) # 64, 100
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples))) # 8

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)) # 100, 100

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_matplot_img(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    self.result_dir + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')
        #save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #            self.result_dir + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read [{}], counter [{}]".format(ckpt_name,counter))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
