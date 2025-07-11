#agent.py：自定义`MahjongGBAgent` 基类，按顺序处理玩家在麻将游戏中观察到的事件，并在每个决策点提取状态特征。同时将网络输出的动作转化为游戏事件。

#+ feature.py：自定义`FeatureAgent` 类，继承自 `MahjongGBAgent`，根据麻将规则处理每个决策点的所有可执行动作，并生成相应的特征表示。

#+ model.py：包含强化学习策略网络、随机策略以及其他可能用于训练和评估的模型。

#+ env.py：自定义环境类`MahjongGBEnv`，继承自 `gym.Env`。该类模拟麻将游戏环境，管理智能体与游戏环境之间的交互，包括状态转换和奖励计算，
#另外有稠密奖励环境类`MahjongGBEnv_Dense_Reward`与打印对局信息（便于调试）的环境类`MahjongGBEnv_Print`。

#+ ppo.py：近端策略优化（PPO）算法实现。

#+ replay_buffer.py：优先经验回放算法实现。

#+ train_ppo_expert_opponents.py：主要脚本，使用近端策略优化（PPO）算法，通过单一智能体对抗专家策略的方式训练强化学习智能体，对手策略为专家策略。

# \_\_main\_\_.py：该脚本用于与 Botzone 平台交互。它处理 Botzone 的输入，通过 `Agent` 类将其转化为状态特征，并将这些特征输入模型以获取动作，
#再将这些动作格式化为 Botzone 兼容的输出格式。不建议修改此文件；然而，可以通过阅读代码来更好地理解 `Agent` 类的接口以及它与 Botzone 的集成方式。


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from env import MahjongGBEnv
from feature import FeatureAgent #根据麻将规则处理每个决策点的所有可执行动作，并生成相应的特征表示
from model import ResnetModel
from torch import nn
from collections import defaultdict
import random
import os
import logging
import matplotlib.pyplot as plt
import warnings

# 计算每个时间步的回报和优势
def compute_returns_and_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    returns = []
    advantages = []
    G = 0
    A = 0

    for i in reversed(range(len(rewards))):
        if dones[i]:
            G = 0
            A = 0

        G = rewards[i] + gamma * G
        returns.insert(0, G)

        if i < len(values) - 1:
            next_value = values[i + 1]
        else:
            next_value = 0

        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        A = delta + gamma * lam * A * (1 - dones[i])
        advantages.insert(0, A)

    return returns, advantages


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    returns = []
    gae = 0
    next_value = 0

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
        returns.insert(0, gae + values[i])

    return returns, advantages


# PPO 更新过程
def ppo_update(policy_network, optimizer, old_log_probs, states, actions, returns, advantages, action_masks,
               clip_param):
    returns = returns.detach()
    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach()

    dataset = TensorDataset(states, actions, returns, old_log_probs, advantages, action_masks)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy_losses = []
    value_losses = []

    for epoch in range(4):
        for batch in dataloader:
            state_batch, action_batch, return_batch, old_log_prob_batch, advantage_batch, action_mask_batch = batch

            input_dict = {'is_training': True, 'obs': {'observation': state_batch, 'action_mask': action_mask_batch}}
            action_logits, value = policy_network(input_dict)
            action_prob = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_prob)
            log_prob = action_dist.log_prob(action_batch)

            ###
            # TODO: 在此处补全 policy_loss 和 value_loss 的计算
            # 提示：
            # 1. 计算 `ratio`，即新策略的概率和旧策略概率的比值。
            # 2. 使用 `clip_param` 对 `ratio` 进行剪辑，并计算 PPO 损失函数中的裁剪损失。
            # 3. 计算 policy_loss，这是损失的策略部分。
            # 4. 计算 value_loss，这是预测值函数和实际回报之间的均方误差。
            ###

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()  # 确保没有重复调用 backward
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)


def reinforce_update(policy_network, optimizer, old_log_probs, states, actions, returns, advantages, action_masks,
               clip_param):
    returns = returns.detach()
    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach()

    dataset = TensorDataset(states, actions, returns, old_log_probs, advantages, action_masks)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy_losses = []
    value_losses = []

    for epoch in range(4):
        for batch in dataloader:
            state_batch, action_batch, return_batch, old_log_prob_batch, advantage_batch, action_mask_batch = batch

            input_dict = {'is_training': True, 'obs': {'observation': state_batch, 'action_mask': action_mask_batch}}
            action_logits, value = policy_network(input_dict)
            action_prob = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_prob)
            log_prob = action_dist.log_prob(action_batch)

            policy_loss = -(log_prob * return_batch).mean()
            value_loss = policy_loss

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()  # 确保没有重复调用 backward
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)
