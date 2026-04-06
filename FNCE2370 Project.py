import numpy as np
import pandas as pd

# -----------------------------
# Dummy inputs
# -----------------------------
mu = np.array([0.0004, 0.0001, 0.0002])
Sigma = np.array([
    [0.00010, 0.00002, 0.00001],
    [0.00002, 0.00005, 0.000015],
    [0.00001, 0.000015, 0.00008]
])

gamma = 3.0
alpha = 0.05
beta = 0.0005
rho = 0.0
n_train = 50000
seed = 42
grid_step = 0.05

np.random.seed(seed)

# -----------------------------
# List all feasible weights
# long-only + full investment
# w1 + w2 + w3 = 1, wi >= 0
# -----------------------------
def generate_weight_grid(step=0.05):
    values = np.round(np.arange(0, 1 + step, step), 50)
    weights = []

    for w1 in values:
        for w2 in values:
            w3 = round(1 - w1 - w2, 10)
            if w3 >= 0:
                weights.append([w1, w2, w3])

    return np.array(weights)

actions = generate_weight_grid(grid_step)

# optional: save all possible weight combinations
weights_df = pd.DataFrame(actions, columns=["w1", "w2", "w3"])
weights_df.to_csv("feasible_weights.csv", index=False)

# -----------------------------
# Simulate one return vector
# -----------------------------
def simulate_return(mu, Sigma):
    return np.random.multivariate_normal(mu, Sigma)

# -----------------------------
# Reward function
# R = rp - gamma/2 * rp^2
# -----------------------------
def get_reward(w, r, gamma):
    rp = np.dot(w, r)
    reward = rp - 0.5 * gamma * (rp ** 2)
    return reward

# -----------------------------
# Stateless Q-learning
# Q is over actions only
# -----------------------------
Q = np.zeros(len(actions))

for t in range(1, n_train + 1):
    epsilon = np.exp(-beta * t)

    if np.random.rand() < epsilon:
        a_idx = np.random.randint(len(actions))
    else:
        a_idx = np.argmax(Q)

    w = actions[a_idx]
    r_t = simulate_return(mu, Sigma)
    reward = get_reward(w, r_t, gamma)

    Q[a_idx] = Q[a_idx] + alpha * (reward + rho * np.max(Q) - Q[a_idx])

# -----------------------------
# Learned best portfolio
# -----------------------------
best_idx = np.argmax(Q)
best_weights = actions[best_idx]

print("Best portfolio weights:")
print(best_weights)

# optional: save ranked actions
q_table = pd.DataFrame(actions, columns=["w1", "w2", "w3"])
q_table["Q_value"] = Q
q_table = q_table.sort_values("Q_value", ascending=False)
q_table.to_csv("q_values_ranked.csv", index=False)