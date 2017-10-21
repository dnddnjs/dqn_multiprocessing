import sys
import gym
import random
import numpy as np
from time import sleep
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from multiprocessing import Process, Queue


# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.batch_size = 64

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        self.epsilon *= self.epsilon_decay
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])


def actor(q1, q2):
    print('start process 1')
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)
    scores = []

    for e in range(100000):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            if q2.qsize() > 0:
                while q2.qsize() > 1:
                    q2.get()
                model = q2.get()
                agent.model.set_weights(model)

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or score == 499 else -100

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            q1.put([state, action, reward, next_state, done])

            score += reward
            state = next_state

            if done:
                sleep(0.1)
                score = score if score == 500 else score + 100
                print("episode:", e, "  score:", score, "  epsilon:", agent.epsilon)
                scores.append(score)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.model.save_weights("./save_model/cartpole_mp.h5")
                    q1.put(None)
                    sys.exit()


def learner(q1, q2):
    print('start process 2')
    replay_memory = deque(maxlen=3000)
    agent = DQNAgent(4, 2)
    stop = False

    for count in range(100000):
        sleep(0.005)

        while q1.qsize() > 0:
            sample = q1.get()
            if sample is None:
                stop = True
            replay_memory.append(sample)

        if stop:
            break

        if len(replay_memory) > 1000:
            mini_batch = random.sample(replay_memory, agent.batch_size)

            states = np.zeros((agent.batch_size, agent.state_size))
            next_states = np.zeros((agent.batch_size, agent.state_size))
            actions, rewards, dones = [], [], []

            for i in range(agent.batch_size):
                states[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_states[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])

            # 현재 상태에 대한 모델의 큐함수
            # 다음 상태에 대한 타깃 모델의 큐함수
            target = agent.model.predict(states)
            target_val = agent.target_model.predict(next_states)

            # 벨만 최적 방정식을 이용한 업데이트 타깃
            for i in range(agent.batch_size):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + agent.discount_factor * (
                        np.amax(target_val[i]))

            agent.model.fit(states, target, batch_size=agent.batch_size,
                            epochs=1, verbose=0)
            model = agent.model.get_weights()
            q2.put(model)

            if (count % 50) == 0:
                print('update target model')
                agent.target_model.set_weights(agent.model.get_weights())


if __name__ == '__main__':
    memory = Queue()
    model = Queue()
    process_one = Process(target=actor, args=(memory, model))
    process_two = Process(target=learner, args=(memory, model))
    process_one.start()
    process_two.start()

    memory.close()
    model.close()
    memory.join_thread()
    model.join_thread()

    process_one.join()
    process_two.join()