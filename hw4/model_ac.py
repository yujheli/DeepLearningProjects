import tensorflow as tf
from opt import *

class GAN:
    def __init__(self, image_size, class_size, embedding_size, noise_size, g_channel_size, d_channel_size, batch_size):
        self.image_size = image_size
        self.class_size = class_size   
        self.embedding_size = embedding_size
        self.noise_size = noise_size
        self.g_channel_size = g_channel_size
        self.d_channel_size = d_channel_size
        self.batch_size = batch_size

    def buildTrainModel(self):
        #placeholder
        t_real_image = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 3], name="real_image")
        t_wrong_image = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 3], name="wrong_image")
        t_right_class = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.class_size], name="right_class")
        t_wrong_class = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.class_size], name="wrong_class")


        t_noise = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.noise_size], name="noise")
        #generator
        t_fake_image = self.generator(t_noise, t_right_class, reuse=False, is_training=True)
        #discriminator
        real_image_logits, real_image_labels = self.discriminator(t_real_image, t_right_class, reuse=False, is_training=True)
        wrong_image_logits, wrong_image_labels = self.discriminator(t_wrong_image, t_right_class, reuse=True, is_training=True)
        fake_right_logits, fake_right_labels = self.discriminator(t_fake_image, t_right_class, reuse=True, is_training=True)
        fake_wrong_logits, fake_wrong_labels = self.discriminator(t_fake_image, t_wrong_class, reuse=True, is_training=True)

        #loss
        t_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_right_logits, labels=tf.ones_like(fake_right_labels)))
        
        d_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_image_logits, labels=tf.ones_like(real_image_labels)))
        d_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_image_logits, labels=tf.zeros_like(wrong_image_labels)))
        d_loss_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_right_logits, labels=tf.zeros_like(fake_right_labels)))
        d_loss_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_wrong_logits, labels=tf.zeros_like(fake_wrong_labels)))

        t_d_loss = d_loss_1 + d_loss_2 + d_loss_3 + d_loss_4
        #trainable_variable
        t_g_variable = [ variable for variable in tf.trainable_variables() if "g_" in variable.name ]
        t_d_variable = [ variable for variable in tf.trainable_variables() if "d_" in variable.name ]
        #return tensor
        t_input_tensor = {
            "real_image": t_real_image,
            "wrong_image": t_wrong_image,
            "r_class": t_right_class,
            "w_class": t_wrong_class,
            "noise": t_noise
        }
        t_output_tensor = {
            "fake_image": t_fake_image
        }
        t_loss_tensor = {
            "g_loss": t_g_loss,
            "d_loss": t_d_loss
        }
        t_variable_tensor = {
            "g_variable": t_g_variable,
            "d_variable": t_d_variable
        }
        check_tensor = {
            "d_loss_1": d_loss_1,
            "d_loss_2": d_loss_2,
            "d_loss_3": d_loss_3,
            "real_image_logits": real_image_logits,
            "wrong_image_logits": wrong_image_logits
            # "fake_image_logits": fake_image_logits
        }
        return t_input_tensor, t_output_tensor, t_loss_tensor, t_variable_tensor, check_tensor

    def buildTestModel(self):
        #placeholder
        t_caption = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.class_size], name="caption")
        t_noise = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.noise_size], name="noise")
        #generator
        t_fake_image = self.generator(t_noise, t_caption, reuse=False, is_training=False)
        #return tensor
        t_input_tensor = {
            "caption": t_caption,
            "noise": t_noise            
        }   
        t_output_tensor = {
            "fake_image": t_fake_image
        }
        return t_input_tensor, t_output_tensor

    def generator(self, noise, class_label, reuse=False, is_training=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        #prepare size
        image_size_16, image_size_8, image_size_4, image_size_2 = int(self.image_size / 16), int(self.image_size / 8), int(self.image_size / 4), int(self.image_size / 2)
        g_channel_size_8, g_channel_size_4, g_channel_size_2 = self.g_channel_size * 8, self.g_channel_size * 4, self.g_channel_size * 2
        #concat noise with caption
        # embed = fullyConnectedLayer(class_label,output_dim=self.embedding_size,name="g_embedding")
        # embed = tf.maximum(embed, 0.2 * embed)

        embed_concat = tf.concat([noise, class_label], 1) 
        h_0 = fullyConnectedLayer(embed_concat, output_dim=image_size_16 * image_size_16 * g_channel_size_8,name="g_fc_0")
        h_0 = tf.reshape(h_0, [-1, image_size_16, image_size_16, g_channel_size_8])
        h_0 = batchNormLayer(h_0, is_training=is_training, name="g_bn_0")
        h_0 = tf.nn.relu(h_0)
        #up-sampling
        h_1 = transConvolutionLayer(h_0, filter_shape=[5, 5, g_channel_size_4, h_0.get_shape()[-1]], stride_shape=[1, 2, 2, 1],
            output_shape=[self.batch_size, image_size_8, image_size_8, g_channel_size_4],
            name="g_tc_1")

        h_1 = batchNormLayer(h_1, is_training=is_training, name="g_bn_1")
        h_1 = tf.nn.relu(h_1)

        h_2 = transConvolutionLayer(h_1, filter_shape=[5, 5, g_channel_size_2, h_1.get_shape()[-1]], stride_shape=[1, 2, 2, 1], output_shape=[self.batch_size, image_size_4, image_size_4, g_channel_size_2],
            name="g_tc_2")
        h_2 = batchNormLayer(h_2, is_training=is_training, name="g_bn_2")
        h_2 = tf.nn.relu(h_2)

        h_3 = transConvolutionLayer(h_2, filter_shape=[5, 5, self.g_channel_size, h_2.get_shape()[-1]], stride_shape=[1, 2, 2, 1], output_shape=[self.batch_size, image_size_2, image_size_2, self.g_channel_size], 
            name="g_tc_3")
        h_3 = batchNormLayer(h_3, is_training=is_training, name="g_bn_3")
        h_3 = tf.nn.relu(h_3)

        h_4 = transConvolutionLayer(h_3, filter_shape=[5, 5, 3, h_3.get_shape()[-1]], stride_shape=[1, 2, 2, 1], output_shape=[self.batch_size, self.image_size, self.image_size, 3], 
            name="g_tc_4")

        #return (tf.tanh(h_4)/2. + 0.5)
        return tf.tanh(h_4)


    def discriminator(self, image, class_label, reuse=False, is_training=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        #prepare size
        d_channel_size_2, d_channel_size_4, d_channel_size_8 = self.d_channel_size * 2, self.d_channel_size * 4, self.d_channel_size * 8
        #down-sample
        h_0 = convolutionLayer(image, filter_shape=[5, 5, image.get_shape()[-1], self.d_channel_size], stride_shape=[1, 2, 2, 1], 
            name="d_c_0")
        h_0 = tf.maximum(h_0, 0.2 * h_0)

        h_1 = convolutionLayer(h_0, filter_shape=[5, 5, h_0.get_shape()[-1], d_channel_size_2], stride_shape=[1, 2, 2, 1], 
            name="d_c_1")
        h_1 = batchNormLayer(h_1, is_training=is_training, name="d_bn_1")
        h_1 = tf.maximum(h_1, 0.2 * h_1)

        h_2 = convolutionLayer(h_1, filter_shape=[5, 5, h_1.get_shape()[-1], d_channel_size_4], stride_shape=[1, 2, 2, 1],
            name="d_c_2")
        h_2 = batchNormLayer(h_2, is_training=is_training, name="d_bn_2")
        h_2 = tf.maximum(h_2, 0.2 * h_2)

        h_3 = convolutionLayer(h_2, filter_shape=[5, 5, h_2.get_shape()[-1], d_channel_size_8], stride_shape=[1, 2, 2, 1],
            name="d_c_3")
        h_3 = batchNormLayer(h_3, is_training=is_training, name="d_bn_3")
        h_3 = tf.maximum(h_3, 0.2 * h_3)
        #concat h_3 with caption
        # embed = fullyConnectedLayer(caption, output_dim=self.embedding_size, name="d_embedding")
        # embed = tf.maximum(embed, 0.2 * embed)
        embed = tf.expand_dims(class_label, 1)
        embed = tf.expand_dims(embed, 2)
        tile_embed = tf.tile(embed, [1, 4, 4, 1])   
        concat_embed = tf.concat([h_3, tile_embed], 3)
        h_4 = convolutionLayer(concat_embed, filter_shape=[1, 1, concat_embed.get_shape()[-1], d_channel_size_8], stride_shape=[1, 1, 1, 1], name="d_c_4")
        h_4 = batchNormLayer(h_4, is_training=is_training, name="d_bn_4")
        h_4 = tf.maximum(h_4, 0.2 * h_4)

        h_5 = fullyConnectedLayer(tf.reshape(h_4, [self.batch_size, -1]), output_dim=1, name="d_fc_5")

        return h_5, tf.nn.sigmoid(h_5)
