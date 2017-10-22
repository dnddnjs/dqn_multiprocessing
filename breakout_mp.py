import sys
import gym
import random
import numpy as np
from time import sleep
import tensorflow as tf
from keras import backend as K
from collections import deque
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from skimage.transform import resize
from skimage.color import rgb2gray
from multiprocessing import Process, Queue


class DQNAgent:
    def __init__(self, action_size):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 500000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.discount_factor = 0.99
        # 리플레이 메모리, 최대 크기 400000
        self.no_op_steps = 30
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_mp', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


def actor(q1, q2, q3):
    print('start process 1')
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(100000):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            if q2.qsize() > 0:
                while q2.qsize() > 1:
                    q2.get()
                model = q2.get()
                agent.model.set_weights(model)

            action = agent.get_action(history)

            # 1: stay, 2: left, 3: right
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)

            q1.put([history, action, reward, next_history, done])

            score += reward

            if dead:
                dead = False
            else:
                history = next_history

            if global_step > 50000:
                print('waiting for process 2, now q size is ', q1.qsize())
                while q1.qsize() > 100:
                    sleep(0.0001)

            if done:
                print("episode:", e, "  score:", score, "  epsilon:",
                      agent.epsilon, "  global_step:", global_step,
                      "  average_q:", agent.avg_q_max / float(step))

                agent.avg_q_max = 0
                scores.append(score)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(10, len(scores)):]) > 100:
                    agent.model.save_weights("./save_model/breakout_mp.h5")
                    q3.put(True)
                    sys.exit()


def learner(q1, q2, q3):
    print('start process 2')
    replay_memory = deque(maxlen=200000)
    agent = DQNAgent(action_size=3)
    count = 0

    while True:
        count += 1
        while q1.qsize() > 0:
            sample = q1.get()
            replay_memory.append(sample)

        if q3.qsize() > 0:
            sys.exit()

        if len(replay_memory) > 50000:
            mini_batch = random.sample(replay_memory, agent.batch_size)

            history = np.zeros((agent.batch_size, agent.state_size[0],
                                agent.state_size[1], agent.state_size[2]))
            next_history = np.zeros((agent.batch_size, agent.state_size[0],
                                     agent.state_size[1], agent.state_size[2]))
            target = np.zeros((agent.batch_size,))
            action, reward, dead = [], [], []

            for i in range(agent.batch_size):
                history[i] = np.float32(mini_batch[i][0] / 255.)
                next_history[i] = np.float32(mini_batch[i][3] / 255.)
                action.append(mini_batch[i][1])
                reward.append(mini_batch[i][2])
                dead.append(mini_batch[i][4])

            target_value = agent.target_model.predict(next_history)

            for i in range(agent.batch_size):
                if dead[i]:
                    target[i] = reward[i]
                else:
                    target[i] = reward[i] + agent.discount_factor * \
                                            np.amax(target_value[i])

            loss = agent.optimizer([history, action, target])

            model = agent.model.get_weights()
            q2.put(model)

            # update per 1000 train is approximately 10000 step in env
            if (count % 500) == 0:
                print('update target model')
                agent.target_model.set_weights(agent.model.get_weights())


if __name__ == '__main__':
    memory = Queue()
    model = Queue()
    end = Queue()
    process_one = Process(target=actor, args=(memory, model, end))
    process_two = Process(target=learner, args=(memory, model, end))
    process_one.start()
    process_two.start()

    memory.close()
    model.close()
    end.close()
    memory.join_thread()
    model.join_thread()
    end.close()

    process_one.join()
    process_two.join()