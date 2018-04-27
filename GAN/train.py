import os
import sys
import random
import tensorflow as tf
from utility import *
from parameter import *
from model_ac import *
from tqdm import tqdm
import numpy as np



fixed_rows = 11
fixed_cols = 12
fixed_size = fixed_rows * fixed_cols
fixed_noise = np.random.normal(size=(fixed_size, noise_size)).astype('float32')
fixed_text = np.zeros((fixed_size, ne))
for i in range(fixed_rows):
    for j in range(fixed_cols):
        fixed_text[i*fixed_cols+j][j] = 1
        fixed_text[i*fixed_cols+j][fixed_cols+i] = 1
print(fixed_text)


def train():
    #get text_image info
    # sample_training_info_list = readSampleInfo(sample_training_info_path)
    loss_fd = open('GAN_ac_loss.txt', 'w')

    if REDUCE_DATASET:
        labels = load_labels()
        train_X, train_y = load_filtered_images_augmentation(labels)
    else:
        train_X = load_images_from_float32()
        train_y = load_labels()




    #model
    model = GAN(
        image_size=image_size,
        class_size=ne,
        embedding_size=embedding_size,
        noise_size=noise_size,
        g_channel_size=g_channel_size,
        d_channel_size=d_channel_size,
        batch_size=batch_size
    )
    #build train model
    with tf.variable_scope("GAN"):
        input_tensor, output_tensor, loss_tensor, variable_tensor, check_tensor = model.buildTrainModel()
    #optimizer
    g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss_tensor['g_loss'], var_list=variable_tensor['g_variable'])
    d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss_tensor['d_loss'], var_list=variable_tensor['d_variable'])
    #session and saver
    session = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=250)
    if restore_flag:
        saver.restore(session, model_ac_dir + "AC_a-" + str(restore_version))
    else:
        session.run(tf.global_variables_initializer())
    #makedirs
    if not os.path.exists(model_ac_dir):
        os.makedirs(model_ac_dir)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    #epoch
    for epoch in range(num_epoch):
        # random.shuffle(sample_training_info_list)
        if REDUCE_DATASET:
            labels = load_labels()
            train_X, train_y = load_filtered_images_augmentation(labels)
        else:
            train_X = load_images_from_float32()
            train_y = load_labels()


        total_num = train_X.shape[0]
        p = np.random.permutation(total_num)
        train_X, train_y = train_X[p], train_y[p]

        if WRONG_LABELS_ARE_ZEROS:
            train_X_wrong, train_y_wrong = np.roll(train_X, 1, axis=0), np.zeros_like(train_y)
        else:
            train_X_wrong, train_y_wrong = np.roll(train_X, 1, axis=0), np.roll(train_y, 1, axis=0)



        for batch in tqdm(range(total_num // batch_size)):
            # current_train_data = sample_training_info_list[batch * batch_size : (batch+1) * batch_size]
            # real_image, wrong_image, caption, noise, image_file = getTrainData(current_train_data)
            real_data = train_X[batch*batch_size:(batch+1)*batch_size]
            wrong_data = train_X_wrong[batch*batch_size:(batch+1)*batch_size]
            right_text_batch = train_y[batch*batch_size:(batch+1)*batch_size]
            wrong_text_batch = train_y_wrong[batch*batch_size:(batch+1)*batch_size]


            noise = np.random.normal(size=(batch_size, noise_size))
            fixed_noise = np.random.normal(size=(fixed_size, noise_size)).astype('float32')


            feed_dict = {
                input_tensor["real_image"]: real_data,
                input_tensor["wrong_image"]: wrong_data,
                input_tensor["r_class"]: right_text_batch,
                input_tensor["w_class"]: wrong_text_batch,
                input_tensor["noise"]: noise
            }           
            g_fetch_dict = {
                "g_optimizer": g_optimizer,
                "g_loss": loss_tensor["g_loss"],
                "fake_image": output_tensor["fake_image"]
            }
            d_fetch_dict = {
                "d_optimizer": d_optimizer,
                "d_loss": loss_tensor["d_loss"],
                "fake_image": output_tensor["fake_image"],
                "d_loss_1": check_tensor["d_loss_1"],
                "d_loss_2": check_tensor["d_loss_2"],
                "d_loss_3": check_tensor["d_loss_3"]
            }
            

            d_track_dict = session.run(d_fetch_dict, feed_dict=feed_dict)
            g_track_dict = session.run(g_fetch_dict, feed_dict=feed_dict)
            g_track_dict = session.run(g_fetch_dict, feed_dict=feed_dict)
            print("\rBatchID: {0}, G losses: {1}, D losses: {2}".format(batch, g_track_dict["g_loss"], d_track_dict["d_loss"]))
            # sys.stdout.flush()
            loss_fd.write("Epoch: {0}, BatchID: {1}, G losses: {2}, D losses: {3} \n".format(epoch, batch, g_track_dict["g_loss"], d_track_dict["d_loss"]))


            # if (batch % save_num_batch) == 0:
            #     test_images = np.concatenate([fixed_text,fixed_text])[:192]
            #     noise = np.random.normal(size=(192, noise_size)).astype('float32')

            #     test_ = []
            #     for i in range(3):

            #         test_feed_dict = {
            #             input_tensor["r_class"]: test_images[i*batch_size:(i+1)*batch_size],
            #             input_tensor["noise"]: noise[i*batch_size:(i+1)*batch_size]
            #         }
            #         test_g_fetch_dict = {
            #             "fake_image": output_tensor["fake_image"]
            #         }

            #         # fixed_noise = np.random.normal(size=(fixed_size, noise_size)).astype('float32')
            #         test_track_dict = session.run(test_g_fetch_dict, feed_dict=test_feed_dict)
            #         test_.append(test_track_dict["fake_image"])


            #     show_and_save(np.concatenate(test_)[:132], fixed_rows,'{0}-{1}.png'.format(epoch, batch))

                


        print("\nEpochID: {0}, Saving Model...\n".format(epoch))

        test_images = np.concatenate([fixed_text,fixed_text])[:192]
        noise = np.random.normal(size=(192, noise_size)).astype('float32')

        test_ = []
        for i in range(3):

            test_feed_dict = {
                input_tensor["r_class"]: test_images[i*batch_size:(i+1)*batch_size],
                input_tensor["noise"]: noise[i*batch_size:(i+1)*batch_size]
            }
            test_g_fetch_dict = {
                "fake_image": output_tensor["fake_image"]
            }

            # fixed_noise = np.random.normal(size=(fixed_size, noise_size)).astype('float32')
            test_track_dict = session.run(test_g_fetch_dict, feed_dict=test_feed_dict)
            test_.append(test_track_dict["fake_image"])


        show_and_save(np.concatenate(test_)[:132], fixed_rows,'{0}-final.png'.format(epoch))

        # test_track_dict = session.run(test_g_fetch_dict, feed_dict=test_feed_dict)
        # show_and_save(test_track_dict["fake_image"], fixed_rows,'{0}-final.png'.format(epoch))
        # saver.save(session, model_dir, global_step=epoch)
        if (epoch % save_num_epoch) == 0 :
            saver.save(session, model_ac_dir+'AC_a', global_step=epoch)

    loss_fd.close()

if __name__ == '__main__':
    train()
