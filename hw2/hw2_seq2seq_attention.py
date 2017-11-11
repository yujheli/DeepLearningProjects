import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
from keras.preprocessing import sequence
# import matplotlib.pyplot as plt

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        # with tf.device("/gpu:0"):
        self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='We')

        #self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        # self.attention_W = tf.Variable( tf.random_uniform([n_video_lstm_step, n_lstm_steps], -0.1, 0.1), name='attention_W')
        # self.attention_b = tf.Variable( tf.zeros([n_lstm_steps]), name='attention_b')

        self.attenW1 = tf.Variable(tf.random_uniform([self.lstm1.state_size, self.dim_hidden], -0.1, 0.1), name='attenW1')
        self.attenW2 = tf.Variable(tf.random_uniform([self.dim_hidden, self.dim_hidden], -0.1, 0.1), name='attenW2')
        self.V = tf.Variable(tf.ones([1]),name='V')


        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')


    def build_model(self):
        #tf.get_variable_scope().reuse_variables()
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):

                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1],1), state2)
                ############################# Encode Output for Attention Based Model #################################
                if i==0:
                    attention_concate = output2
                else:
                    attention_concate = tf.concat([attention_concate,output2],1)


        # attention_flat = tf.reshape(attention_concate,[-1,self.n_video_lstm_step])
        # attention_embed =  tf.nn.xw_plus_b( attention_flat, self.attention_W, self.attention_b )
        # attention_embed = tf.reshape(attention_embed, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        ############################# Decoding Stage ##############################################################
        for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
            #if i == 0:
            #    current_embed = tf.zeros([self.batch_size, self.dim_hidden])
            #else:
            # with tf.device("/gpu:0"):
            current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
        ############################# Implementation of Attention Based Model ######################################
                a = tf.tile(tf.expand_dims(tf.matmul(state1, self.attenW1),1),[1,self.n_video_lstm_step,1])
                b = tf.reshape(tf.matmul(tf.reshape(attention_concate,[-1, self.dim_hidden]), self.attenW2),[self.batch_size, self.n_video_lstm_step,self.dim_hidden])
                c = tf.nn.softmax(tf.tanh(tf.add(a,b))*self.V)*tf.reshape(attention_concate,[self.batch_size, self.n_video_lstm_step,self.dim_hidden])


                output1, state1 = self.lstm1(tf.reduce_sum(c,1), state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1],1), state2)

            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat( [indices, labels],1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss
            
            

        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1],1), state2)
                ############################# Encode Output for Attention Based Model #################################
                if i==0:
                    attention_concate = output2
                else:
                    attention_concate = tf.concat([attention_concate,output2],1)

        for i in range(0, self.n_caption_lstm_step):
            tf.get_variable_scope().reuse_variables()

            if i == 0:
                # with tf.device('/gpu:0'):
                current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
            ############################# Implementation of Attention Based Model ######################################

                a = tf.tile(tf.expand_dims(tf.matmul(state1, self.attenW1),1),[1,self.n_video_lstm_step,1])
                b = tf.reshape(tf.matmul(tf.reshape(attention_concate,[-1, self.dim_hidden]), self.attenW2),[1, self.n_video_lstm_step,self.dim_hidden])
                c = tf.nn.softmax(tf.tanh(tf.add(a,b))*self.V)*tf.reshape(attention_concate,[1, self.n_video_lstm_step,self.dim_hidden])
                


                output1, state1 = self.lstm1(tf.reduce_sum(c,1), state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1],1), state2)

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            # with tf.device("/gpu:0"):
            current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
            current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


#=====================================================================================
# Global Parameters
#=====================================================================================


model_path = './models'

#=======================================================================================
# Train Parameters
#=======================================================================================
dim_image = 4096
dim_hidden= 1000

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

n_epochs = 100000
batch_size = 50
learning_rate = 0.01


def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
   # print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print ('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector

def train(mode):


    print("Preparing for the data......")

    train_name = pd.read_json(sys.argv[2]+'/training_label.json')['id']
    train_data = []
    for i in train_name:
        train_data.append(np.load(sys.argv[2]+'/training_data/feat/'+i+'.npy'))
    train_data = np.array(train_data)


    train_raw_label = pd.read_json(sys.argv[2]+'/training_label.json')['caption']
    test_raw_label = pd.read_json(sys.argv[2]+'/testing_label.json')['caption']
    
    train_captions = []
    train_label = []
    for i in train_raw_label.index:
        train_captions+=train_raw_label[i]
        #used for generating label
        train_label.append(train_raw_label[i])
    test_captions = []
    for i in test_raw_label.index:
        test_captions+=test_raw_label[i]
    #Get the captions 
    print("Preprocess for the labels......")
    captions_list = list(train_captions) + list(test_captions)
    captions = np.asarray(captions_list, dtype=np.object)

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)

    #I change the directory name
    np.save("wordtoix", wordtoix)
    np.save('ixtoword', ixtoword)
    np.save("bias_init_vector", bias_init_vector)
    print("Setup the model......")


    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=20)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)

    if mode==1:
        saver.restore(sess, sys.argv[3])
    else:
        sess.run(tf.global_variables_initializer())
        


    loss_fd = open('loss.txt', 'w')
    loss_to_draw = []

    #shuffle
    # idx = np.random.permutation(len(train_data))
    # train_data,train_label = np.array(train_data)[idx], np.array(train_label)[idx]


    print("Starting epoch......")

    for epoch in range(0, n_epochs):
        idx = np.random.permutation(len(train_data))
        train_data,train_label = np.array(train_data)[idx], np.array(train_label)[idx]

        loss_to_draw_epoch = []
        train_label_cur = [i[np.random.choice(len(i))] for i in train_label]
        # train_label_cur = [i[np.random.choice(len(i))] for i in train_label]


        for start, end in zip(range(0, len(train_data)+1, batch_size),range(batch_size, len(train_data)+1, batch_size)):

            start_time = time.time()
            
            # current_feats_vals = map(lambda vid: np.load(vid), current_videos)
            current_feats_vals = train_data[start:end]

            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
            current_video_masks = np.zeros((batch_size, n_video_lstm_step))

            # for mask and for padding
            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat

                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            # current_captions = current_batch['Description'].values
            current_captions = train_label_cur[start:end]


            current_captions = map(lambda x: '<bos> ' + x, current_captions)
            current_captions = map(lambda x: x.replace('.', ''), current_captions)
            current_captions = map(lambda x: x.replace(',', ''), current_captions)
            current_captions = map(lambda x: x.replace('"', ''), current_captions)
            current_captions = map(lambda x: x.replace('\n', ''), current_captions)
            current_captions = map(lambda x: x.replace('?', ''), current_captions)
            current_captions = map(lambda x: x.replace('!', ''), current_captions)
            current_captions = map(lambda x: x.replace('\\', ''), current_captions)
            current_captions = list(map(lambda x: x.replace('/', ''), current_captions))

            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < n_caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(n_caption_lstm_step-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) ))

            for ind, row in enumerate(current_caption_masks):
                row[:(nonzeros[ind])] = 1

            probs_val = sess.run(tf_probs, feed_dict={
                tf_video:current_feats,
                tf_caption: current_caption_matrix
                })

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })

            print ('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
            loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

        if np.mod(epoch, 100) == 0:
            print ("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model_atte_Jack'), global_step=epoch)

    loss_fd.close()

def test():
    model_path=sys.argv[4]
    #test_raw_label = pd.read_json(sys.argv[2]+'/testing_label.json')['caption']

    
    test_name = np.array(pd.read_csv(sys.argv[2]+'/testing_id.txt',header=None)[0])
    test_data = []
    for i in test_name:
        test_data.append(np.load(sys.argv[2]+'/testing_data/feat/'+i+'.npy'))
    test_data = np.array(test_data)

    ixtoword = pd.Series(np.load('./ixtoword.npy').tolist())

    bias_init_vector = np.load('./bias_init_vector.npy')

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open(sys.argv[3], 'w')
    for idx, video_feat in enumerate(test_data):
        
        video_feat_rec = video_feat.reshape(1,80,4096)
        if video_feat_rec.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat_rec.shape[0], video_feat_rec.shape[1]))
        else:
            continue
            #shape_templete = np.zeros(shape=(1, n_frame_step, 4096), dtype=float )
            #shape_templete[:video_feat.shape[0], :video_feat.shape[1], :video_feat.shape[2]] = video_feat
            #video_feat = shape_templete
            #video_mask = np.ones((video_feat.shape[0], n_frame_step))

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat_rec, video_mask_tf:video_mask})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        #print (test_name[idx],generated_sentence, test_raw_label[idx][0])
        print("testing finished!")
        test_output_txt_fd.write(test_name[idx] + ',')
        test_output_txt_fd.write(generated_sentence + '\n')
    test_output_txt_fd.close()

def test_special():
    model_path=sys.argv[4]

    test_name = ['klteYv1Uv9A_27_33.avi','5YJaS2Eswg0_22_26.avi','UbmZAe5u5FI_132_141.avi','JntMAcTlOF0_50_70.avi','tJHUH9tpqPg_113_118.avi']
    test_data = []
    for i in test_name:
        test_data.append(np.load(sys.argv[2]+'/testing_data/feat/'+i+'.npy'))
    test_data = np.array(test_data)

    ixtoword = pd.Series(np.load('./ixtoword.npy').tolist())

    bias_init_vector = np.load('./bias_init_vector.npy')

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open(sys.argv[3], 'w')
    for idx, video_feat in enumerate(test_data):
        
        video_feat_rec = video_feat.reshape(1,80,4096)
        if video_feat_rec.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat_rec.shape[0], video_feat_rec.shape[1]))
        else:
            continue
            #shape_templete = np.zeros(shape=(1, n_frame_step, 4096), dtype=float )
            #shape_templete[:video_feat.shape[0], :video_feat.shape[1], :video_feat.shape[2]] = video_feat
            #video_feat = shape_templete
            #video_mask = np.ones((video_feat.shape[0], n_frame_step))

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat_rec, video_mask_tf:video_mask})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        print (test_name[idx],generated_sentence)
        test_output_txt_fd.write(test_name[idx] + ',')
        test_output_txt_fd.write(generated_sentence + '\n')
    test_output_txt_fd.close()

if __name__ == '__main__':
    if sys.argv[1]=='--train':
        train(0)
    elif sys.argv[1]=='--test':
        test()
    elif sys.argv[1]=='--train_cont':
        train(1)

    elif sys.argv[1]=='--special':
        test_special()


    

