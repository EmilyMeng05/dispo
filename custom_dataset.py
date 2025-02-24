import gym
import d4rl
import numpy as np

# Paste the CustomD4RLDataset and OfflineDataset class definitions here

# Example usage
if __name__ == "__main__":
    # Initialize the environment
    env = gym.make('hopper-medium-v2')  # Replace with your desired environment

    # Initialize the custom dataset
    dataset = CustomD4RLDataset(env)

    # Sample data
    sampled_data = dataset.sample(num_samples=5)
    print("State:", sampled_data["state"])
    print("Actions:", sampled_data["actions"])
    print("Next State:", sampled_data["next_state"])
    print("Rewards:", sampled_data["rewards"])
    print("Dones:", sampled_data["dones"])

