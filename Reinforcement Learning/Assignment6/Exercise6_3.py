import numpy as np
from CartPole import CartPoleEnv
import matplotlib.pyplot as plt

H = 30
learning_rate = 0.001
gamma = 0.99
decay_rate = 0.99
score_queue_size = 100

D = 4

model = {}
model['W1'] = np.random.randn(H, D)
model['W2'] = np.random.randn(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}


def getForce(self, x, x_dot, theta, theta_dot):
    k = np.random.uniform(0, 1, 4)
    return min(100, max(-100, k[0] * x + k[1] * x_dot + k[2] * theta + k[3] * theta_dot))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    return I


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h = sigmoid(h)  # shape (10, 1)
    logp = np.dot(model['W2'], h)
    p = (logp)  # shape (1)
    # print(p)
    # print('h')
    # print(np.shape(p))
    return p, h


def policy_backward(eph, epdlogp, epx):
    global grad_buffer
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    eph_dot = eph * (1 - eph)
    dW1 = dh * eph_dot
    dW1 = np.dot(dW1.T, epx)

    for k in model: grad_buffer[k] += {'W1': dW1, 'W2': dW2}[k]



env = CartPoleEnv()
env.reset()
observation = env.state
reward_sum, episode_num = 0, 0
xs, hs, dlogps, drs = [], [], [], []
score_queue = []

NUM_EPI = 1000
t = 0

rewards = []
while True:

    state = observation

    act_prob, h = policy_forward(state)

    action = act_prob

    xs.append(state)
    hs.append(h)
    y = action
    dlogps.append(y - act_prob)
    # print(state)
    env.getNextState(action)
    reward = env.getReward()
    observation = env.state
    t = t + 1
    reward_sum += reward

    drs.append(reward)

    if env.isTerminal:

        reward_sum += -(1000 - t)
        rewards.append(reward_sum)
        drs[len(drs) - 1] = -(1000 - t)
        episode_num += 1

        if episode_num > score_queue_size:
            score_queue.append(reward_sum)
            score_queue.pop(0)
        else:
            score_queue.append(reward_sum)

        if episode_num % 100 == 0:
            print('time step when pole falls down: ' + str(t))
            print("episode : " + str(episode_num) + ", reward : " + str(reward_sum) + ", reward_mean : " + str(
                np.mean(score_queue)))

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr

        policy_backward(eph, epdlogp, epx)
        for k, v in model.items():
            g = grad_buffer[k]
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)

        if episode_num % 1000 == 0: print(model)

        reward_sum = 0
        env.reset()
        observation = env.state
        t = 0

        if episode_num == 1000:
            break

fig, ax = plt.subplots()
ax.plot(rewards)
ax.set(xlabel='episodes', ylabel='final reward')
ax.grid()
plt.title('Exercise 6.3')
plt.show()

