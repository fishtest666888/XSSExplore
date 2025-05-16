import preprocess
import numpy as np
from collections import deque
import os
import torch
import argparse
import pickle
from buffer import ReplayBuffer
from utils_rl import save, collect_random
import random
from agent import SAC
import xss_env
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Flatten, LSTM, Conv1D, MaxPool1D
from keras.optimizers import Adam
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mlp_model_dir = "./file/MLP_model"
lstm_model_dir = "./file/LSTM_model"
svm_model_dir = "./file/p_SVM_model"
cnn_model_dir = "./file/Conv_model"


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="SAC", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=540, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size, default: 256")
    args = parser.parse_args()
    return args


# def mlp_model(input_num, dims_num, batch_size):
#     inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
#     layer1 = Dense(100, activation="relu")
#     layer2 = Dense(20, activation="relu")
#     flatten = Flatten()
#     layer3 = Dense(2, activation="softmax", name="Output")
#     optimizer = Adam()
#     model = Sequential()
#     model.add(inputs)
#     model.add(layer1)
#     model.add(Dropout(0.5))
#     model.add(layer2)
#     model.add(Dropout(0.5))
#     model.add(flatten)
#     model.add(layer3)
#     model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
#     model.load_weights(mlp_model_dir, by_name=False)
#     return model
#
#
# def lstm_model(input_num, dims_num, batch_size):
#     inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
#     layer1 = LSTM(128)
#     output = Dense(2, activation="softmax", name="Output")
#     optimizer = Adam()
#     model = Sequential()
#     model.add(inputs)
#     model.add(layer1)
#     model.add(Dropout(0.5))
#     model.add(output)
#     model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
#     model.load_weights(lstm_model_dir, by_name=False)
#     return model
#
#
# def svm_model():
#     file = open(svm_model_dir, 'rb')
#     xss_svm = pickle.load(file)
#     file.close()
#     return xss_svm
#
#
# def cnn_model(input_num, dims_num, batch_size):
#     inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
#     layer1 = Conv1D(64, 3, activation="relu")
#     layer2 = Conv1D(64, 3, activation="relu")
#     layer3 = Conv1D(128, 3, activation="relu")
#     layer4 = Conv1D(128, 3, activation="relu")
#     layer5 = Dense(128, activation="relu")
#     output = Dense(2, activation="softmax", name="Output")
#     optimizer = Adam()
#     model = Sequential()
#     model.add(inputs)
#     model.add(layer1)
#     model.add(layer2)
#     model.add(MaxPool1D(pool_size=2))
#     model.add(Dropout(0.5))
#     model.add(layer3)
#     model.add(layer4)
#     model.add(MaxPool1D(pool_size=2))
#     model.add(Dropout(0.5))
#     model.add(Flatten())
#     model.add(layer5)
#     model.add(Dropout(0.5))
#     model.add(output)
#     model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
#     model.load_weights(cnn_model_dir, by_name=False)
#     return model


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = xss_env.XSS_Moutation()
    env.seed(get_config().seed)
    # env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    steps = 0

    average10 = deque(maxlen=10)
    agent = SAC(state_size=len(env.observation_space), action_size=len(env.action_space), device=device)
    data = preprocess.data_process()
    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
    # MLP_model = mlp_model(532, 128, 1)
    # LSTM_model = lstm_model(532, 128, 1)
    # SVM_model = svm_model()
    CNN_model = cnn_model(532, 128, 1)
    collect_random(env=env, dataset=buffer, data=data, num_samples=400, model=CNN_model)  # 往经验池存数据，用于更新网络参数

    # 先调用检测模型，被检测出来后再变异
    escape_num = 0
    step_write = []
    for i in range(1, config.episodes + 1):
        payload = env.reset(data, i - 1)
        state = np.array(env.observation_space)
        episode_steps = 0
        rewards = 0
        while True:
            action = agent.get_action(state)
            steps += 1
            next_state, reward, new_payload, done = env.step(episode_steps, action, payload, CNN_model)
            buffer.add(state, action, reward, next_state, done)
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
            state = next_state
            rewards += reward
            episode_steps += 1
            payload = new_payload
            if not done or episode_steps == env.max_episode_steps:
                if not done:
                    escape_num += 1
                break

        average10.append(rewards)
        print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, episode_steps))
        step_write.append(episode_steps)
    file = open('cnn.txt', 'w')
    file.write(str(step_write))
    file.close()
    return escape_num


if __name__ == "__main__":
    config = get_config()
    start = time.time()
    ER = train(config)
    end = time.time()
    print("Escape number: ", ER)
    print("Escape rate: ", ER/540)
    print("time: ", end-start)
