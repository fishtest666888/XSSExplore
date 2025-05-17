from gym.utils import seeding
from urllib.parse import quote
import mutation
import time
import random
import numpy as np
import MLP_test
import LSTM_test
import SVM_single
import CNN_test
space = []
for i in range(75):
    space.append(-1)


class XSS_Moutation():
    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # 动作空间，包括可以使用的绕过策略
        self.observation_space = space  # 观察空间，表示实体的状态，[5,2,0,....0]，表示第一步采用策略5，第二步采用策略2，...
        self.seed()
        self.max_episode_steps = 75
        self.payload = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, step, action, payload, model):  # 输入动作，返回当前状态
        _, new_payload = mutation.action_payload(action, payload)
        # self.observation_space[step] = action![](MaxStep.jpg)
        self.observation_space[step] = action
        # done = MLP_test.mlp_test_result(quote(new_payload), model)  # 调用MLP分类器
        # done = LSTM_test.lstm_test_result(quote(new_payload), model)  # 调用LSTM分类器
        # done = SVM_single.svm_test_result(model, quote(new_payload))  # 调用SVM分类器
        done = CNN_test.cnn_test_result(quote(new_payload), model)  # 调用CNN分类器
        if done:  # True 如果被检测出来是恶意样本
            reward = 0
        else:  # 没被检测出来
            # print("绕过:", quote(new_payload))
            reward = 1
        return np.array(self.observation_space), reward, new_payload, done

    def reset(self, data, total_steps):
        if not data:
            raise ValueError("Data list is empty in reset()")
        self.payload = data[total_steps % len(data)]  # 避免越界
        self.observation_space = [-1] * 75
        return self.payload



