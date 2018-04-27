#model
image_size = 64
caption_size = 4800
embedding_size = 256
noise_size = 100
ne = 23
g_channel_size = 64
d_channel_size = 64
batch_size = 64

#optimizer
learning_rate = 0.0002
beta1 = 0.5

#train
restore_flag = False
restore_version = 500


FACE_PATH = 'facesnpy_resized_float32'
OUTPUT_PATH = 'output_cdcgan_onehot'

REDUCE_DATASET=True
WRONG_LABELS_ARE_ZEROS = True


num_epoch = 600
save_num_batch = 50
save_num_epoch = 1
max_to_keep = 300

HAIR_TAGS = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
                 'green hair', 'red hair', 'purple hair', 'pink hair',
                 'blue hair', 'black hair', 'brown hair', 'blonde hair']

EYES_TAGS = ['gray eyes', 'black eyes', 'orange eyes',
                'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                'green eyes', 'brown eyes', 'red eyes', 'blue eyes']





#path
sample_training_info_path = "./info/sample_training_info"
sample_testing_info_path = "./info/sample_testing_info"
model_ac_dir="./models/"+"AC/"

model_dir = "./models/" + "_".join((
		"image_size", str(image_size),
		"caption_size", str(caption_size),
		"embedding_size", str(embedding_size),
		"noise_size", str(noise_size),
		"g_channel_size", str(g_channel_size),
		"d_channel_size", str(d_channel_size),
		"batch_size", str(batch_size),
		"learning_rate", str(learning_rate),
		"beta1", str(beta1)
    )) + "/"
result_training_dir = "./results/training/" + "_".join((
		"image_size", str(image_size),
		"caption_size", str(caption_size),
		"embedding_size", str(embedding_size),
		"noise_size", str(noise_size),
		"g_channel_size", str(g_channel_size),
		"d_channel_size", str(d_channel_size),
		"batch_size", str(batch_size),
		"learning_rate", str(learning_rate),
		"beta1", str(beta1)
    )) + "/"
result_testing_dir = "./results/testing/" + "_".join((
		"image_size", str(image_size),
		"caption_size", str(caption_size),
		"embedding_size", str(embedding_size),
		"noise_size", str(noise_size),
		"g_channel_size", str(g_channel_size),
		"d_channel_size", str(d_channel_size),
		"batch_size", str(batch_size),
		"learning_rate", str(learning_rate),
		"beta1", str(beta1)
    )) + "/"
sample_dir = "./samples/"
result_training_caption_path = result_training_dir + "caption.txt" 
result_testing_caption_path = result_testing_dir + "caption.txt"
