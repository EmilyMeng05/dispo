import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class OfflineDataset(Dataset):
    def __init__(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
    ):
        if len(observations.shape) == 4:
            obs_dtype = np.uint8
        else:
            obs_dtype = np.float32
        self.observations = np.array(observations).astype(obs_dtype)
        self.actions = np.array(actions).astype(np.float32)
        self.rewards = np.array(rewards).astype(np.float32).reshape(-1, 1)
        self.next_observations = np.array(next_observations).astype(obs_dtype)
        self.dones = np.array(dones).astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return dict(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_observations=self.next_observations[idx],
            dones=self.dones[idx],
        )


class D4RLDataset(OfflineDataset):
    def __init__(self, env, num_actions, sequence_length=3):
        # Set num_actions
        self.num_actions = num_actions

        # Load dataset using env.get_dataset()
        dataset = env.get_dataset()

        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        dones = dataset["terminals"]

        # Compute next_observations if not present
        if "next_observations" not in dataset:
            next_observations = np.roll(observations, -1, axis=0)
            # Handle the last observation
            next_observations[-1] = observations[-1]  
        else:
            next_observations = dataset["next_observations"]

        # Handle AntMaze-specific logic
        if "antmaze" in env.spec.id:
            # Compute dense rewards
            goal = np.array(env.target_goal)
            dists_to_goal = np.linalg.norm(
                goal[None] - next_observations[:, :2], axis=-1
            )
            rewards = np.exp(-dists_to_goal / 20)
        elif "kitchen" in env.spec.id:
            # Remove goals from observations
            observations = observations[:, :30]
            next_observations = observations[:, :30]

        super().__init__(observations, actions, rewards, next_observations, dones)

        self.sequence_length = sequence_length
        self.full_dataset = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
            "rewards": rewards,
            "dones": dones,
        }

    def __getitem__(self, idx):
        # Ensure idx is within bounds
        if idx + self.num_actions >= len(self.full_dataset["observations"]):
            idx = len(self.full_dataset["observations"]) - self.num_actions - 1

        # Sample the initial state at index idx
        state = self.full_dataset["observations"][idx]

        # Select actions and next states
        action_indices = np.arange(idx, idx + self.num_actions)
        actions = self.full_dataset["actions"][action_indices]
        next_states = self.full_dataset["next_observations"][action_indices]

        # Pick the last next state
        next_state = next_states[-1]  # Use -1 to get the last element

        # Return the state, actions, and next state
        return dict(
            state=state,
            actions=actions,
            next_state=next_state,
            rewards=self.full_dataset["rewards"][idx + self.num_actions],
            dones=self.full_dataset["dones"][idx + self.num_actions],
        )


class RoboverseDataset(OfflineDataset):
    def __init__(self, env, task, data_dir="data/roboverse"):
        if task == "pickplace-v0":
            prior_data_path = os.path.join(data_dir, "pickplace_prior.npy")
            task_data_path = os.path.join(data_dir, "pickplace_task.npy")
        elif task == "doubledraweropen-v0":
            prior_data_path = os.path.join(data_dir, "closed_drawer_prior.npy")
            task_data_path = os.path.join(data_dir, "drawer_task.npy")
        elif task == "doubledrawercloseopen-v0":
            prior_data_path = os.path.join(data_dir, "blocked_drawer_1_prior.npy")
            task_data_path = os.path.join(data_dir, "drawer_task.npy")
        else:
            raise NotImplementedError("Unsupported roboverse task")

        prior_data = np.load(prior_data_path, allow_pickle=True)
        task_data = np.load(task_data_path, allow_pickle=True)

        full_data = np.concatenate((prior_data, task_data))
        dict_data = {}
        for key in [
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
        ]:
            full_values = []
            for traj in full_data:
                values = traj[key]
                if key == "observations" or key == "next_observations":
                    full_values += [env.observation(obs) for obs in values]
                else:
                    full_values += values
            dict_data[key] = np.array(full_values)

        super().__init__(
            dict_data["observations"],
            dict_data["actions"],
            dict_data["rewards"],
            dict_data["next_observations"],
            dict_data["terminals"],
        )


class AntMazePreferenceDataset(OfflineDataset):
    def __init__(self, env):
        import d4rl

        dataset = d4rl.qlearning_dataset(
            env,
            h5path="data/d4rl/Ant_maze_obstacle_noisy_multistart_True_multigoal_True.hdf5",
        )
        rewards = env.compute_reward(dataset["next_observations"])
        super().__init__(
            dataset["observations"],
            dataset["actions"],
            rewards,
            dataset["next_observations"],
            dataset["terminals"],
        )


class RaMPDataset(OfflineDataset):
    def __init__(self, env, dataset_dir="data/ramp"):
        data_dir = os.path.join(dataset_dir, "HopperEnv-v5", "rand_2048")
        rollout_fns = sorted(glob.glob(os.path.join(data_dir, "*.rollout")))
        
        observations, actions, rewards, next_observations, dones = [], [], [], [], []
        for rollout_fn in rollout_fns:
            rollout = torch.load(rollout_fn)
            obs_dim = rollout["obs"].shape[2]
            action_dim = rollout["action"].shape[2]
            
            # Flatten out episodes
            observations.append(rollout["obs"][:, :-1].reshape(-1, obs_dim))
            actions.append(rollout["action"][:, :-1].reshape(-1, action_dim))
            next_observations.append(rollout["obs"][:, 1:].reshape(-1, obs_dim))
            raw_dones = np.zeros_like(rollout["done"][:, 1:].reshape(-1, 1))
            raw_dones[-1] = 1
            dones.append(raw_dones)

            # Relabel rewards by querying the environment
            env_rewards = np.array([env.compute_reward(o) for o in next_observations[-1]])
            rewards.append(env_rewards)

        # Initialize parent class
        super().__init__(
            np.concatenate(observations),
            np.concatenate(actions),
            np.concatenate(rewards),
            np.concatenate(next_observations),
            np.concatenate(dones),
        )

    def __getitem__(self, idx):
        # sample a state
        state = self.full_dataset["observations"][idx]

        # randomly select multiple actions and their corresponding next states
        action_indices = np.random.choice(len(self.full_dataset["actions"]), self.num_actions, replace=False)
        actions = self.full_dataset["actions"][action_indices]
        next_states = self.full_dataset["next_observations"][action_indices]
        rewards = self.full_dataset["rewards"][action_indices]
        dones = self.full_dataset["dones"][action_indices]

        return dict(
            state=state,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            dones=dones,
        )