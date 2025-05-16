import torch
import random
import numpy as np


def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")


def collect_random(env, dataset, data, num_samples, model):
    n = random.randint(0, 500)
    payload = env.reset(data, n)
    state = np.array(env.observation_space)
    reset_num = 0
    for i in range(num_samples):
        random.seed(i+1000)
        action = random.randint(0, 14)
        print(i, action)
        next_state, reward, new_payload, done = env.step(reset_num, action, payload, model)
        dataset.add(state, action, reward, next_state, done)
        payload = new_payload
        state = next_state
        reset_num += 1
        if not done:  # False 如果没被检测出来
            reset_num = 0
            n = random.randint(0, 500)
            payload = env.reset(data, n)
            state = env.observation_space
        if reset_num == 50:
            reset_num = 0
            n = random.randint(0, 500)
            payload = env.reset(data, n)
            state = env.observation_space
