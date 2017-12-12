from agent_dir.agent import Agent

import numpy as np
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, Reshape
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import scipy


def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)




        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            # self.model = load_model('pong_model.h5')
        #else:
            #self.model = load_model('pong_model.h5')

        # def get_session(gpu_fraction):
        #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        #     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
        # Kt.set_session(get_session(0.1))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        set_session(tf.Session(config=config))

        
        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.prev_x = None
        self.cur_x = None
        self.learning_rate = 0.001




    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_x = None
        self.cur_x = None

        self.action_size = self.env.action_space.n
        self.state_size = (80,80,1)

        self.model = self._build_model_2()
        self.load('pong_model_weights_t.h5')

        # model = Sequential()
        # model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        # model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
        #                         activation='relu', init='he_uniform'))
        # model.add(Flatten())
        # model.add(Dense(64, activation='relu', init='he_uniform'))
        # model.add(Dense(32, activation='relu', init='he_uniform'))
        # model.add(Dense(self.action_size, activation='softmax'))
        # self.learning_rate = 0.001

        # opt = Adam(lr=self.learning_rate)
        # model.compile(loss='categorical_crossentropy', optimizer='adam')
        # model.load_weights('pong.h5')
        # self.model = model
        

        

        


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.action_size = self.env.action_space.n
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.total_loss = .0
        
        

        state = self.env.reset()
        # self.state_size = state.shape
        self.state_size = (80,80,1)
        #prev_x = None
        score = 0
        episode = 0

        self.model = self._build_model_2()
        # self.load('pong_model_weights_4.h5')
        self.load('pong_model_weights_t.h5')
        self.model.summary()


        # self.model.compile(loss='categorical_crossentropy',
        #               optimizer=RMSprop(lr=self.learning_rate))


        log_fd = open('log_pg_3_3.txt', 'w')
        log_fd.write('epoch,score\n')

        while True:
            #self.env.render()

            # self.cur_x = self.preprocess(state)
            # duration += 1
            self.cur_x = prepro(state)
            
            x = self.cur_x - self.prev_x if self.prev_x is not None else np.zeros([80,80,1])
            self.prev_x = self.cur_x

            action, prob = self.act(x)
            state, reward, done, info = self.env.step(action)
            score += reward
            self.remember(x, action, prob, reward)

            if done:
                episode += 1
                self._train()
                print('Episode: %d - Score: %f - loss %f.' % (episode, score, self.total_loss))
                log_fd.write(str(episode)+','+str(score)+'\n')
                score = 0
                # duration = 0
                self.total_loss = .0
                state = self.env.reset()
                prev_x = None
                if episode > 1 and episode % 10 == 0:
                    self.save('pong_model_weights_t.h5')
        


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        

        self.cur_x = prepro(observation)
            
        input_ob = self.cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.state_size)
        self.prev_x = self.cur_x

        input_ob = input_ob.reshape([1, 80, 80, 1])
        aprob = self.model.predict(input_ob, batch_size=1).flatten()
       
        # prob = aprob / np.sum(aprob)
        # action = np.random.choice(self.action_size, 1, p=prob)[0]
        action = np.argmax(aprob)


        # return self.env.get_random_action()
        return action


    def preprocess(self,I):
        I = I[35:195]
        I = I[::2, ::2, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()


    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        model.add(Conv2D(32, (6, 6), activation='relu', padding='same',kernel_initializer='random_uniform',
                bias_initializer='zeros'))
        #model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=self.state_size))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        #model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        # model.add(Conv2D(64, (3, 3), activation='relu',padding ='same',kernel_initializer='random_uniform',
        #         bias_initializer='zeros'))
        # model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        # model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
        #                         activation='relu', init='he_uniform'))


        model.add(Flatten())
        model.add(Dense(64, activation='relu',kernel_initializer='random_uniform',
                bias_initializer='zeros'))
        model.add(Dense(32, activation='relu',kernel_initializer='random_uniform',
                bias_initializer='zeros'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def _build_model_2(self):
        model = Sequential()
        # model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        model.add(Conv2D(16, (6, 6), strides = (3,3), activation='relu', input_shape=(80,80,1),kernel_initializer='he_uniform'))
        #model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=self.state_size))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        #model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Conv2D(32, (5, 5), strides = (2,2), activation='relu',kernel_initializer='he_uniform'))
        # model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        # model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
        #                         activation='relu', init='he_uniform'))


        model.add(Flatten())
        # model.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))
        # model.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def _build_model_3(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides = (4,4), activation='relu', input_shape=(80,80,1),kernel_initializer='he_uniform'))
        model.add(Conv2D(32, (4, 4), strides = (2,2), activation='relu',kernel_initializer='he_uniform'))

        model.add(Flatten())
        model.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape([1, 80, 80, 1])
        # state = state.reshape([1, state.shape[0],state.shape[1],state.shape[2]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def _train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        X = np.expand_dims(X,axis=-1)
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        #self.model.train_on_batch(X, Y)
        his = self.model.fit(X, Y, batch_size = len(Y), epochs=1, verbose=0)
        self.total_loss = his.history['loss'][0]
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)




