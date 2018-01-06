import os
import sys
import scipy
from utility import *
from parameter import *
from model_ac import *
import pandas as pd


testing_file = sys.argv[1]
model_dir_ = sys.argv[2]
# seed_num = int(sys.argv[3])

def generate():
    #get text_image info
    # sample_testing_info_list = readSampleInfo(sample_testing_info_path)
    sample_info = pd.read_csv(testing_file,header=None)

    #model
    model = GAN(
        image_size=image_size,
        class_size=ne,
        embedding_size=embedding_size,
        noise_size=noise_size,
        g_channel_size=g_channel_size,
        d_channel_size=d_channel_size,
        batch_size=1
    )   
    #build test model
    with tf.variable_scope("GAN"):
        input_tensor, output_tensor = model.buildTestModel()
    #session and saver
    session = tf.InteractiveSession()
    saver = tf.train.Saver()
    # saver.restore(session, model_dir + "AC-" + str(int(model_version)*10 + restore_version))
    saver.restore(session, model_dir_)


    #makedirs
    # if not os.path.exists(result_testing_dir):
    #   os.makedirs(result_testing_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    for batch in range(len(sample_info) // 1):


        id_ = sample_info[0][batch]
        caption = convert_tags_for_hw4(sample_info[1][batch])

        g_fetch_dict = {
            "fake_image": output_tensor["fake_image"]
        }
        # current_test_data = sample_testing_info_list[batch : (batch+1) * 1]

        # caption, noise, image_file = getTestData(current_test_data)

        np.random.seed(2)
        seed_list = [3, 9, 10, 20, 27]
        for i in range(5):
            
            np.random.seed(seed_list[i])
            noise = np.asarray(np.random.uniform(-1, 1, [1, noise_size]), dtype=np.float32)

            feed_dict = {
                input_tensor['caption']: caption.reshape(1,caption.shape[0]),
                input_tensor['noise']: noise
            }
            print("Generating id:{0} sample id:{1}".format(id_,i))
            
            g_track_dict = session.run(g_fetch_dict, feed_dict=feed_dict)

            # sys.stdout.write("\rBatchID: {0}, Saving Image...".format(batch))
            # sys.stdout.flush()
            
            int_X = ((np.array(g_track_dict["fake_image"][0])+1)/2*255).clip(0,255).astype('uint8')
            scipy.misc.imsave(sample_dir +'sample_'+ str(id_) + '_' + str(i) + ".jpg", int_X)

if __name__ == '__main__':
    generate()
