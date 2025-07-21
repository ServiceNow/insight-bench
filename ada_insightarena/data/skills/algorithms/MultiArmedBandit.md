

# Multi-Armed Bandit Pipeline



Required Imports


``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
```



Data Preparation



``` python
def data_preparation(num_arms=10, seed=2023):
    np.random.seed(seed)
    reward_probs = np.random.uniform(0, 1, num_arms)
    best_arm_index = np.argmax(reward_probs)
    arm_labels = [str(x) for x in range(num_arms)]
    return reward_probs, best_arm_index, arm_labels
```


Data Manipulation



``` python
def data_manipulation(reward_probs):
    # Placeholder for any data manipulation needed in future.
    # Currently, reward probabilities are used directly without transformation.
    return reward_probs
```



Multi-Armed Bandit Implementation



``` python
# Model Implementation (Multi-Armed Bandit)
def model_implementation(num_arms, reward_probs):
    class MultiArmedBandit:
        def __init__(self, num_arm, reward_probs):
            self.num_arm = num_arm
            self.reward_probs = reward_probs

        def run_iteration(self, iters, time_steps):
            rewards_hists = []
            explored_cnts = []
            for i in range(iters):
                reward_hist, explored_cnt = self.run_episode(time_steps)
                rewards_hists.append(reward_hist)
                explored_cnts.append(explored_cnt)
            avg_reward_hist = np.mean(rewards_hists, axis=0)
            std_reward_hist = np.std(rewards_hists, axis=0)
            avg_explored_cnt = np.mean(explored_cnts, axis=0)
            std_explored_cnt = np.std(explored_cnts, axis=0)
            return avg_reward_hist, std_reward_hist, avg_explored_cnt, std_explored_cnt

        def run_episode(self, time_steps):
            rewards = []
            exp_reward_prob = np.zeros(self.num_arm)
            explored_cnt = np.zeros(self.num_arm)
            for trial in range(time_steps):
                arm, reward = self.step()
                exp_reward_prob, explored_cnt = self.update(arm, reward, exp_reward_prob, explored_cnt)
                rewards.append(reward)
            return rewards, explored_cnt

        def step(self):
            arm = np.random.choice(range(self.num_arm))
            reward = np.random.choice([0, 1], p=[1-self.reward_probs[arm], self.reward_probs[arm]])
            return arm, reward

        def update(self, arm, reward, exp_reward_prob, explored_cnt):
            explored_cnt[arm] += 1
            exp_reward_prob[arm] = exp_reward_prob[arm] + (1/explored_cnt[arm] * (reward - exp_reward_prob[arm]))
            return exp_reward_prob, explored_cnt

        def plot_algorithm_performance(self, avg_reward_hist, std_reward_hist):
            plt.figure(figsize=(10, 5))
            plt.plot(avg_reward_hist, label='Average Reward')
            plt.fill_between(range(len(avg_reward_hist)),
                             avg_reward_hist - std_reward_hist,
                             avg_reward_hist + std_reward_hist,
                             color='b', alpha=0.2)
            plt.xlabel('Time Step')
            plt.ylabel('Average Reward')
            plt.title('Performance of the Multi-Armed Bandit')
            plt.legend()
            plt.show()

    return MultiArmedBandit
```



Visualization


``` python
# Visualization
def visualize_reward_probabilities(arm_labels, reward_probs, best_arm_index):
    plt.figure(figsize=(10, 5))
    barlist = plt.bar(arm_labels, reward_probs, color='gray')
    barlist[best_arm_index].set_color('r')
    plt.xlabel('Arm index', fontsize=15)
    plt.ylabel('Reward probability', fontsize=15)
    plt.title('Reward Probabilities of Each Arm')
    plt.show()
```



Evaluate Results



``` python
def evaluate_results(bandit, num_iters=100, time_steps=1000):
    avg_reward_hist, std_reward_hist, avg_explored_cnt, std_explored_cnt = bandit.run_iteration(iters=num_iters, time_steps=time_steps)
    bandit.plot_algorithm_performance(avg_reward_hist, std_reward_hist)
    return avg_reward_hist, std_reward_hist, avg_explored_cnt, std_explored_cnt
```



Save Results



``` python
# Save Results
def save_results(avg_reward_hist, avg_explored_cnt):
    results_df = pd.DataFrame({
        'Avg Reward': avg_reward_hist,
        'Avg Explored Count': avg_explored_cnt
    })
    results_df.to_csv('multi_armed_bandit_results.csv', index=False)
```



Main Execution



``` python
# Main Execution
def main():
    num_arms = 10
    reward_probs, best_arm_index, arm_labels = data_preparation(num_arms)
    reward_probs = data_manipulation(reward_probs)
    MultiArmedBandit = model_implementation(num_arms, reward_probs)

    bandit = MultiArmedBandit(num_arms, reward_probs)
    visualize_reward_probabilities(arm_labels, reward_probs, best_arm_index)
    avg_reward_hist, std_reward_hist, avg_explored_cnt, std_explored_cnt = evaluate_results(bandit)
    save_results(avg_reward_hist, avg_explored_cnt)

if __name__ == "__main__":
    main()
```

