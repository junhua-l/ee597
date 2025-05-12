from multi_user_network import env_network
from drqn import QNetwork, Memory
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import deque
import os
import tensorflow as tf
import time

# ---------------------- 创建保存目录 ----------------------
os.makedirs('figures', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# ---------------------- 超参 & 初始化 ----------------------
TIME_SLOTS     = 15000
NUM_CHANNELS   = 5
NUM_USERS      = 6
ATTEMPT_PROB   = 1

memory_size    = 1000
batch_size     = 6
pretrain_length= batch_size
hidden_size    = 128
learning_rate  = 0.0001
explore_start  = 0.02
explore_stop   = 0.01
decay_rate     = 0.0001
gamma          = 0.9
step_size      = 1 + 2 + 2
state_size     = 2 * (NUM_CHANNELS + 1)
action_size    = NUM_CHANNELS + 1
alpha          = 0
beta           = 1

record_interval = 100    # 每隔 N 步记录一次 loss
loss_history    = []

# 重置 TF 图
tf.reset_default_graph()

# 环境与网络
env    = env_network(NUM_USERS, NUM_CHANNELS, ATTEMPT_PROB)
mainQN = QNetwork(
    name='main',
    hidden_size=hidden_size,
    learning_rate=learning_rate,
    step_size=step_size,
    state_size=state_size,
    action_size=action_size
)
memory       = Memory(max_size=memory_size)
history_input= deque(maxlen=step_size)

saver = tf.train.Saver()
sess  = tf.Session()
sess.run(tf.global_variables_initializer())

# 存储变量
total_rewards  = []
cum_r          = [0]
cum_collision  = [0]


# ---------------------- 辅助函数 ----------------------

def one_hot(num, length):
    assert 0 <= num < length
    v = np.zeros([length], np.int32)
    v[num] = 1
    return v

def state_generator(action, obs):
    input_vector = []
    for user_i in range(action.size):
        iv = one_hot(action[user_i], NUM_CHANNELS + 1)
        iv = np.append(iv, obs[-1])                # channel residual vector
        iv = np.append(iv, int(obs[user_i][0]))    # ACK
        input_vector.append(iv)
    return input_vector

# 预填充 memory
action = env.sample()
obs    = env.step(action)
state  = state_generator(action, obs)
for _ in range(pretrain_length * step_size * 5):
    action     = env.sample()
    obs        = env.step(action)
    next_state = state_generator(action, obs)
    reward     = [i[1] for i in obs[:NUM_USERS]]
    memory.add((state, action, reward, next_state))
    state = next_state
    history_input.append(state)

# 用户级数据提取函数
def get_states_user(batch):
    states = []
    for user in range(NUM_USERS):
        states_per_user = []
        for each in batch:
            states_per_batch = []
            for step_i in each:
                states_per_step = step_i[0][user]
                states_per_batch.append(states_per_step)
            states_per_user.append(states_per_batch)
        states.append(states_per_user)
    return np.array(states)

def get_actions_user(batch):
    actions = []
    for user in range(NUM_USERS):
        actions_per_user = []
        for each in batch:
            actions_per_batch = []
            for step_i in each:
                actions_per_step = step_i[1][user]
                actions_per_batch.append(actions_per_step)
            actions_per_user.append(actions_per_batch)
        actions.append(actions_per_user)
    return np.array(actions)

def get_rewards_user(batch):
    rewards = []
    for user in range(NUM_USERS):
        rewards_per_user = []
        for each in batch:
            rewards_per_batch = []
            for step_i in each:
                rewards_per_step = step_i[2][user]
                rewards_per_batch.append(rewards_per_step)
            rewards_per_user.append(rewards_per_batch)
        rewards.append(rewards_per_user)
    return np.array(rewards)

def get_next_states_user(batch):
    next_states = []
    for user in range(NUM_USERS):
        next_states_per_user = []
        for each in batch:
            next_states_per_batch = []
            for step_i in each:
                next_states_per_step = step_i[3][user]
                next_states_per_batch.append(next_states_per_step)
            next_states_per_user.append(next_states_per_batch)
        next_states.append(next_states_per_user)
    return np.array(next_states)


# ---------------------- 主训练循环 ----------------------

for time_step in range(TIME_SLOTS):
    # 调整 beta
    if time_step % 50 == 0 and time_step < 5000:
        beta -= 0.001

    explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * time_step)

    # 选择动作
    if explore_p > np.random.rand():
        action = env.sample()
    else:
        action = np.zeros([NUM_USERS], dtype=np.int32)
        state_vector = np.array(history_input)
        for u in range(NUM_USERS):
            feed = {mainQN.inputs_: state_vector[:, u].reshape(1, step_size, state_size)}
            Qs   = sess.run(mainQN.output, feed_dict=feed)
            prob1= (1 - alpha) * np.exp(beta * Qs)
            prob = prob1 / np.sum(np.exp(beta * Qs)) + alpha / (NUM_CHANNELS + 1)
            action[u] = np.argmax(prob, axis=1)

    obs        = env.step(action)
    next_state = state_generator(action, obs)
    reward     = [i[1] for i in obs[:NUM_USERS]]
    sum_r      = np.sum(reward)
    collision  = NUM_CHANNELS - sum_r

    cum_r.append(cum_r[-1] + sum_r)
    cum_collision.append(cum_collision[-1] + collision)

    # cooperative reward
    for i in range(len(reward)):
        if reward[i] > 0:
            reward[i] = sum_r

    total_rewards.append(sum_r)
    memory.add((state, action, reward, next_state))
    state = next_state
    history_input.append(state)

    # 训练步骤
    batch          = memory.sample(batch_size, step_size)
    states         = get_states_user(batch)
    actions_arr    = get_actions_user(batch)
    rewards_arr    = get_rewards_user(batch)
    next_states    = get_next_states_user(batch)

    states      = np.reshape(states,      [-1, states.shape[2],     states.shape[3]])
    actions_arr = np.reshape(actions_arr, [-1, actions_arr.shape[2]])
    rewards_arr = np.reshape(rewards_arr, [-1, rewards_arr.shape[2]])
    next_states = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])

    target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
    targets   = rewards_arr[:, -1] + gamma * np.max(target_Qs, axis=1)

    loss, _ = sess.run([mainQN.loss, mainQN.opt],
                       feed_dict={
                           mainQN.inputs_:   states,
                           mainQN.targetQs_: targets,
                           mainQN.actions_:  actions_arr[:, -1]
                       })

    # 记录 loss
    if time_step % record_interval == 0:
        loss_history.append(loss)

    # 每 5000 步保存并绘图
    if time_step % 5000 == 4999:
        # 累积碰撞 + 奖励
        fig, axes = plt.subplots(2, 1, figsize=(5, 10))  # 宽度改为 5
        for ax in axes:
            ax.tick_params(axis='both', labelsize=14)
        axes[0].plot(cum_collision, 'r-')
        axes[0].set_xlabel('Time Slot', fontsize=16)
        axes[0].set_ylabel('Cumulative Collision', fontsize=16)
        axes[1].plot(cum_r, 'r-')
        axes[1].set_xlabel('Time Slot', fontsize=16)
        axes[1].set_ylabel('Cumulative Reward', fontsize=16)
        fig.tight_layout()
        fig.savefig(f'figures/collision_reward_until_{time_step}.png', dpi=300)
        plt.close(fig)

        # Loss 曲线
        fig2 = plt.figure(figsize=(5, 4))  # 宽度改为 5
        ax2 = fig2.gca()
        ax2.plot(np.arange(len(loss_history)) * record_interval, loss_history, '-')
        ax2.set_xlabel('Training Step', fontsize=16)
        ax2.set_ylabel('Loss', fontsize=16)
        ax2.set_title('Training Loss History', fontsize=18)
        ax2.tick_params(axis='both', labelsize=14)
        fig2.tight_layout()
        fig2.savefig(f'figures/loss_until_{time_step}.png', dpi=300)
        plt.close(fig2)

        # 可选：三合一图
        fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 12))  # 宽度改为 5
        for ax in (ax1, ax2, ax3):
            ax.tick_params(axis='both', labelsize=14)
        ax1.plot(cum_collision, 'r-')
        ax1.set_ylabel('Cumulative Collision', fontsize=16)
        ax2.plot(cum_r, 'r-')
        ax2.set_ylabel('Cumulative Reward', fontsize=16)
        ax3.plot(np.arange(len(loss_history)) * record_interval, loss_history, '-')
        ax3.set_xlabel('Training Step', fontsize=16)
        ax3.set_ylabel('Loss', fontsize=16)
        fig3.tight_layout()
        fig3.savefig(f'figures/all_metrics_until_{time_step}.png', dpi=300)
        plt.close(fig3)


        # 重置统计
        total_rewards.clear()
        cum_r         = [0]
        cum_collision = [0]
        loss_history.clear()

        saver.save(sess, f'checkpoints/dqn_multi-user_{time_step}.ckpt')
