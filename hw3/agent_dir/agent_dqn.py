from agent_dir.agent import Agent
import numpy as np
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, Reshape
from keras.layers import Conv2D, MaxPooling2D, Lambda
from skimage.color import rgb2gray
from collections import deque
import random
from keras import backend as K
import keras.backend.tensorflow_backend as Kt
import tensorflow as tf


def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here

            print('loading trained model')

        # def get_session(gpu_fraction):
        #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        #     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
        # Kt.set_session(get_session(0.4))
        #self.model = load_model('Breakout_model.h5')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        Kt.set_session(tf.Session(config=config))


        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.INITIAL_REPLAY_SIZE = 20000  # Number of steps to populate the replay memory before training starts
        #self.NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training



    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_x = None
        self.state_size = (84,84,4)
        self.action_size = self.env.action_space.n
        self.epsilon = 0.1
        self.do = 0
        self.learning_rate = 0.01
        self.model = self._build_model3()
      

        # self.load('Breakout_model_weights_e2.h5')
        self.load('Breakout_model_weights_duel.h5')



    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _huber_loss_2(self, target, prediction):
        # sqrt(1+error^2)-1
        mask = K.one_hot(K.argmax(target,axis=-1),self.action_size)
        #pred = K.sum(prediction*mask, axis=-1)
        error = prediction - target
        error*= mask
        return K.mean(K.sqrt(1+K.square(K.sum(error,axis=-1)))-1, axis=-1)




    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_x = None
        self.cur_x = None

        EXPLORATION_STEPS = 1000000

        

        self.state_size = (84,84,4)
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate

        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.model = self._build_model2()
        #self.load('Breakout_model_weights_e2.h5')

        self.target_model = self._build_model2()
        self.update_target_model()
        self.batch_size = 32
        self.TRAIN_INTERVAL = 4
        self.INITIAL_REPLAY_SIZE = 3000
        self.STATE_LENGTH = 4
        self.TARGET_INTERVAL = 3000

        self.epsilon_step = (self.epsilon-self.epsilon_min)/EXPLORATION_STEPS

        state = self.env.reset()
        self.prev_x = state[:]
        # cur = self.get_initial_state(state, self.prev_x)
        # self.state_size = state.shape
        #self.state_size = 80 * 80
        #prev_x = None
        score = 0
        episode = 0
        self.total_loss = .0
        self.total_q = .0
        self.QQ = [.0]
        self.dur_q = 1

        #self.model = self._build_model()
        # self.model.compile(loss='mse',optimizer='adam')
        self.model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        # self.model.compile(loss=self._huber_loss_2,
        #               optimizer=RMSprop(lr=self.learning_rate))
        
        self.model.summary()

        log_fd = open('log_dgn_sim.txt', 'w')
        log_fd.write('epoch,score,loss,Q\n')
        train_flag = 0
        duration = 0
        while True:
            #self.env.render()

            
            # if self.prev_x is None:
            #     self.prev_x = state[:]
                # self.prev_x = np.zeros(self.state_size)


            # self.cur_x = state[:]
            # x = self.preprocess(state, self.prev_x)
            # x = np.array([x])
            
            # x = self.cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.state_size)
            # self.prev_x = state
            # for _ in range(random.randint(1, 15)):
            #     #last_observation = observation
            #     state, _, _, _ = self.env.step(0)  # Do nothing


            action  = self.act(state)
            duration += 1 
            cur = state[:]
            state, reward, done, info = self.env.step(action)
            # next_state = self.preprocess(state, self.prev_x)
            # if reward<0:
            #     print(reward)
            score += reward
            # if done:
            #     reward = -1
            self.remember(cur, action, reward, state, done)

            train_flag += 1

            #self.remember(x, action, prob, reward)
            if train_flag % self.TARGET_INTERVAL==0:
                self.update_target_model()
              

            if len(self.memory) > self.INITIAL_REPLAY_SIZE:
                if train_flag % self.TRAIN_INTERVAL==0:
                    # self.replay(self.batch_size)
                    self.replay_ddqn2(self.batch_size)
        


            if done:
                episode += 1
                if self.dur_q>1:
                    self.dur_q -=1
                
                print('Episode: %d - Score: %f - epsilon %f - loss %f -Q %f.' % (episode, score, self.epsilon, self.total_loss/(duration/4), np.max(self.QQ)))
                # print('Episode: %d - Score: %f - epsilon %f - loss %f -Q %f.' % (episode, score, self.epsilon, self.total_loss/duration/4, self.total_q/self.dur_q))
                log_fd.write(str(episode)+','+str(score)+','+str(self.total_loss/(duration/4))+','+ str(self.total_q/self.dur_q)+'\n')
                # self.update_target_model()
                score = 0
                self.total_loss = .0
                duration = 0
                self.dur_q = 1
                self.total_q = .0
                state = self.env.reset()
                self.QQ = [.0]
                # self.prev_x = state[:]
                # cur = self.get_initial_state(state, self.prev_x)
                # prev_x = None

                # if self.epsilon > self.epsilon_min and len(self.memory) > self.INITIAL_REPLAY_SIZE:
                #     self.epsilon *= self.epsilon_decay

                if episode > 1 and episode % 10 == 0:
                    self.save('Breakout_model_weights_sim.h5')
                    #self.model.save('Breakout_model.h5')




    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # self.prev_x = None
        # self.cur_x = None

        # if self.prev_x is None:
        #     self.prev_x = observation
        #         # self.prev_x = np.zeros(self.state_size)

        #     #self.cur_x = state[:]
        # x = self.preprocess(observation, self.prev_x)
        # # x = np.array([x])
        
        # # x = self.cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.state_size)
        # self.prev_x = observation


        # action  = self.act(observation)
        # if self.do_nothing == None:
        #     self.do_nothing = random.randint(1, 15)

        # if self.do_nothing>0:
        #     self.do_nothing -= 1
        #     return 0
        # else:
        # if self.do%4 == 0:
        #     a = 2
        #     self.do+=1
        # else:
        #     a = 3
        # return a
        # a = random.choice([1,2,3])
        # print (a)
        # return a
        self.epsilon = 0.005
        if np.random.rand() <= self.epsilon:

            return random.randrange(self.action_size)

        act_values = self.model.predict(observation.reshape(1,84,84,4))
        
        #print (np.argmax(act_values[0]))
        # return 0
        return np.argmax(act_values[0])  # returns action








        # return self.env.get_random_action()
        # return action


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),activation='relu', input_shape=(84, 84, 4),kernel_initializer='random_uniform',
                bias_initializer='zeros'))
        # model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Conv2D(64, (4, 4), strides=(2, 2),activation='relu',kernel_initializer='random_uniform',
                bias_initializer='zeros'))
        # model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1),activation='relu',kernel_initializer='random_uniform',
                bias_initializer='zeros'))
        # model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Flatten())
        model.add(Dense(512, activation='relu',kernel_initializer='random_uniform',
                bias_initializer='zeros'))
        # model.add(Dense(self.num_actions))

        model.add(Dense(self.action_size,kernel_initializer='random_uniform',
                bias_initializer='zeros'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_model2(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),activation='relu', input_shape=(84, 84, 4),kernel_initializer='he_uniform'))
        # model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Conv2D(64, (4, 4), strides=(2, 2),activation='relu',kernel_initializer='he_uniform'))
        # model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1),activation='relu',kernel_initializer='he_uniform'))
        # model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Flatten())
        model.add(Dense(512, activation='relu',kernel_initializer='he_uniform'))
        # model.add(Dense(self.num_actions))

        model.add(Dense(self.action_size,kernel_initializer='he_uniform',
                bias_initializer='zeros'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def _build_model3(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),activation='relu', input_shape=(84, 84, 4),kernel_initializer='he_uniform'))
        # model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Conv2D(64, (4, 4), strides=(2, 2),activation='relu',kernel_initializer='he_uniform'))
        # model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1),activation='relu',kernel_initializer='he_uniform'))
        # model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
        model.add(Flatten())
        model.add(Dense(512, activation='relu',kernel_initializer='he_uniform'))
        # model.add(Dense(self.num_actions))

        model.add(Dense(self.action_size+1,kernel_initializer='he_uniform'))
        model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), 
                         output_shape=(self.action_size,)))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model





       

    def remember(self, state, action, reward, next_state, done):
        # next_ = np.append(state[:,:,1:],next_state,axis=-1)

        # self.memory.append((state, action, reward, next_state, done))
        self.memory.append((state, action, reward, next_state, done))




    def act(self, state):
        if self.epsilon > self.epsilon_min and len(self.memory) > self.INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step
       

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        self.dur_q += 1
        act_values = self.model.predict(state.reshape(1,84,84,4))

        # ##Softmax choose
        # prob = softmax(act_values.flatten())
        
        # action = np.random.choice(self.action_size, 1, p=prob)[0]

        self.total_q += np.max(act_values[0])
        self.QQ.append(np.max(act_values[0]))
        
        #print (np.argmax(act_values[0]))
        # return action
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        dones = np.array(dones)+0

        A_q_s = self.model.predict(next_states, batch_size=batch_size)

        # target_q_s = self.target_model.predict(next_states, batch_size=batch_size)
        max_target_q_s = rewards + (1-dones) * self.gamma * np.max(A_q_s,axis=1)
        target = self.model.predict(states, batch_size=batch_size)

        target[range(batch_size), actions] = max_target_q_s
            # target = np.zeros((1,self.action_size))

        his = self.model.fit(states, target, epochs=1, verbose=0)
        self.total_loss += his.history['loss'][0]



        # for state, action, reward, next_state, done in minibatch:
        #     target = reward
        #     if not done:
        #         target = (reward + self.gamma *
        #                   np.amax(self.model.predict(next_state.reshape(1,84,84,4))[0]))

        #     target_f = self.model.predict(state.reshape(1,84,84,4))
        #     # target_f = np.zeros((1,self.action_size))
        #     target_f[0][action] = target
        #     his = self.model.fit(state.reshape(1,84,84,4), target_f, epochs=1, verbose=0)
        #     self.total_loss += his.history['loss'][0]





        # if self.epsilon > self.epsilon_min:
           
        #     self.epsilon *= self.epsilon_decay

    def replay_ddqn(self, batch_size):
        X = []
        Y = []
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state.reshape(1,84,84,4))
            # target = np.zeros((1,self.action_size))

            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state.reshape(1,84,84,4))[0]
                t = self.target_model.predict(next_state.reshape(1,84,84,4))[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
                # his = self.model.fit(state.reshape(1,84,84,4), target, epochs=1, verbose=0)
            Y.append(target[0])
            X.append(state)
        his = self.model.fit(np.array(X), np.array(Y), epochs=1, verbose=0)
        self.total_loss += his.history['loss'][0]
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
    def replay_ddqn2(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        Y = []
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        dones = np.array(dones)+0

        A_q_s = self.model.predict(next_states, batch_size=batch_size)

        target_q_s = self.target_model.predict(next_states, batch_size=batch_size)
        max_target_q_s = rewards + (1-dones) * self.gamma * target_q_s[range(batch_size),np.argmax(A_q_s, axis=1)]



        target = self.model.predict(states, batch_size=batch_size)

        target[range(batch_size), actions] = max_target_q_s
            # target = np.zeros((1,self.action_size))

        his = self.model.fit(states, target, epochs=1, verbose=0)
        self.total_loss += his.history['loss'][0]
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


    def preprocess(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = rgb2gray(processed_observation)
        return np.reshape(processed_observation, (84, 84, 1))


    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = rgb2gray(processed_observation)
        state = [processed_observation for _ in range(4)]
        return np.stack(state, axis=-1)

