import functools
import os

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from omegaconf import OmegaConf
from orbax import checkpoint
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from environments import make_env_and_dataset
from models.nets import ConditionalUnet1D
from models.sampling import get_pc_sampler
from models.sde_lib import VPSDE
from models.utils import TrainState, EMATrainState, get_loss_fn
from user_int import get_user_input


def build_models(config, env, dataset, rng, num_actions):
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = np.prod(env.action_space.shape)
    feat_dim = env.feat_dim

    # Values for initializing models
    init_obs = jnp.array([env.observation_space.sample()])
    init_act = jnp.array([env.action_space.sample()])
    init_psi = jnp.ones((1, feat_dim))
    init_t = jnp.array([0.0])

    # Define scaler and inverse scaler
    psi_min = dataset.feature_min / (1 - config.gamma)
    psi_max = dataset.feature_max / (1 - config.gamma)
    psi_range = psi_max - psi_min
    psi_scaler = lambda x: (x - psi_min) / psi_range * 2 - 1
    psi_inv_scaler = lambda x: (x + 1) / 2 * psi_range + psi_min

    # Define cosine lr scheduler with warmup
    lr_fn = optax.warmup_cosine_decay_schedule(
        0, config.model.lr, config.model.warmup_steps, config.training.num_steps
    )

    # Initialize psi model
    psi_def = ConditionalUnet1D(
        output_dim=feat_dim,
        global_cond_dim=obs_dim,
        num_actions=num_actions,
        embed_dim=config.model.embed_dim,
        embed_type=config.model.embed_type,
    )
    rng, psi_rng = jax.random.split(rng)
    psi_params = psi_def.init(psi_rng, init_psi, init_t, init_obs)["params"]
    psi = EMATrainState.create(
        model_def=psi_def,
        params=psi_params,
        ema_rate=config.model.ema_rate,
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_fn, weight_decay=config.model.wd),
        ),
    )
    psi_sde = VPSDE(
        beta_min=config.model.beta_min,
        beta_max=config.model.beta_max,
        N=config.model.num_steps,
    )
    psi_loss_fn = get_loss_fn(
        psi_sde,
        psi.model_def,
        psi_scaler,
        config.model.continuous,
    )
    psi_sampler = get_pc_sampler(
        psi_sde,
        psi.model_def,
        (feat_dim,),
        config.sampling.predictor,
        config.sampling.corrector,
        psi_inv_scaler,
        config.model.continuous,
        config.sampling.n_inference_steps,
        eta=config.sampling.eta,
    )

    # Initialize policy
    policy_def = ConditionalUnet1D(
        # output multiple action
        output_dim=act_dim * num_actions,  
        global_cond_dim=obs_dim + feat_dim,
        num_actions=num_actions,
        embed_dim=config.model.embed_dim,
        embed_type=config.model.embed_type,
    )
    rng, policy_rng = jax.random.split(rng)
    policy_params = policy_def.init(
        policy_rng, init_act, init_t, jnp.concatenate([init_obs, init_psi], -1)
    )["params"]
    policy = EMATrainState.create(
        model_def=policy_def,
        params=policy_params,
        ema_rate=config.model.ema_rate,
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_fn, weight_decay=config.model.wd),
        ),
    )
    policy_sde = VPSDE(
        beta_min=config.model.beta_min,
        beta_max=config.model.beta_max,
        N=config.model.num_steps,
    )
    policy_loss_fn = get_loss_fn(
        policy_sde,
        policy.model_def,
        lambda x: x,
        config.model.continuous,
    )
    policy_sampler = get_pc_sampler(
        policy_sde,
        policy.model_def,
        (num_actions, act_dim),
        config.sampling.predictor,
        config.sampling.corrector,
        lambda x: x,
        config.model.continuous,
        config.sampling.n_inference_steps,
        eta=config.sampling.eta,
    )

    # Infer reward weights
    w_def = nn.Dense(1, use_bias=False)
    reward_weights = dataset.infer_reward_weights(config.num_reward_samples)
    w_params = {"kernel": reward_weights}
    w = TrainState.create(
        model_def=w_def,
        params=w_params,
    )

    # Build planner
    guidance_fn = (
        lambda w: w.params["kernel"].T * 0.5 * psi_range * config.planning.guidance_coef
    )
    planner = get_planner(
        config.planning.planner,
        guidance_fn,
        config.planning.num_samples,
        config.planning.num_elites,
    )

    return (
        psi,
        psi_sampler,
        psi_loss_fn,
        policy,
        policy_sampler,
        policy_loss_fn,
        w,
        planner,
        rng,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "config",
        "psi_sampler",
        "psi_loss_fn",
        "policy_loss_fn",
    ),
)
def update(
    config,
    rng,
    psi,
    psi_sampler,
    psi_loss_fn,
    policy,
    policy_loss_fn,
    batch,
    num_actions,
):
    # Sample target psi
    rng, sample_rng = jax.random.split(rng)
    next_psi = psi_sampler(psi.ema_params, sample_rng, batch["next_observations"])
    target_psi = batch["features"] + config.gamma * next_psi

    # Update psi
    rng, loss_rng = jax.random.split(rng)
    psi, psi_info = psi.apply_loss_fn(
        loss_fn=psi_loss_fn,
        rng=loss_rng,
        x=target_psi,
        cond=batch["observations"],
        has_aux=True,
    )

    # Update policy
    rng, loss_rng = jax.random.split(rng)
    cond = jnp.concatenate([batch["observations"], target_psi], -1)
    actions = batch["actions"].reshape(-1, num_actions, env.action_space.shape[0])
    policy, policy_info = policy.apply_loss_fn(
        loss_fn=policy_loss_fn,
        rng=loss_rng,
        x=actions,
        cond=cond,
        has_aux=True,
    )

    train_info = {
        "train/psi_loss": psi_info["loss"],
        "train/policy_loss": policy_info["loss"],
    }
    return rng, psi, policy, train_info


def evaluate(config, rng, env, planner, psi, psi_sampler, policy, policy_sampler, w, num_actions):
    # Evaluate online
    obs = env.reset()
    obs = jnp.array(obs[None])
    done = False
    ep_reward, ep_success = 0, 0
    frames = []
    episode_length = 0

    while not done:
        rng, actions, pinfo = planner(
            rng, psi, psi_sampler, policy, policy_sampler, w, obs
        )
        actions = actions.reshape(-1, num_actions, env.action_space.shape[0])

        # Execute all actions in the sequence
        for action in actions[0]:
            next_obs, reward, done, info = env.step(np.array(action))
            ep_reward += reward
            ep_success += info.get("success", 0)
            obs = jnp.array(next_obs[None])
            episode_length += 1

            # Render frame
            frame = env.render(mode="rgb_array", width=128, height=128)
            frames.append(frame)

            # Visualize values
            if config.training.log_psi_video:
                values = w(pinfo["psis"]).sum(-1)
                fig = plt.figure(figsize=(3, 3))
                plt.hist(values, bins=value_bins, density=True)
                plt.ylim([0, 1])
                plt.title("Value")
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                psi_frames.append(data)
                plt.close(fig)

            if done:
                break

    # Compute distance to goal
    goal = np.array(env.target_goal)
    final_position = obs[0, :2]
    distance_to_goal = np.linalg.norm(goal - final_position)

    # Video shape: (T, H, W, C) -> (N, T, C, H, W)
    video = np.stack(frames).transpose(0, 3, 1, 2)[None]
    eval_info = {
        "test/return": ep_reward,
        "test/success": float(ep_success > 0),
        "test/episode_length": episode_length,
        "test/distance_to_goal": distance_to_goal,
        "test/video": wandb.Video(video, fps=30, format="gif"),
    }
    return rng, eval_info


@hydra.main(version_base=None, config_path="configs/", config_name="dispo.yaml")
def train(config):
    # Initialize wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "all-task-rl"),
        group=config.env_id,
        job_type=config.algo,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )

    # Make environment and dataset
    env, dataset = make_env_and_dataset(
        config.env_id, config.seed, config.feat.type, config.feat.dim
    )

    # Get user input for num_actions
    num_actions = get_user_input()

    # Pass num_actions to the dataset
    dataset = D4RLDataset(env, num_actions=num_actions)

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )

    # Round steps to epochs
    num_epochs = config.training.num_steps // len(dataloader)
    num_steps = num_epochs * len(dataloader)
    OmegaConf.update(config, "training.num_steps", num_steps)

    # Define RNG
    rng = jax.random.PRNGKey(config.seed)

    # Build models
    (
        psi,
        psi_sampler,
        psi_loss_fn,
        policy,
        policy_sampler,
        policy_loss_fn,
        w,
        planner,
        rng,
    ) = build_models(config, env, dataset, rng, num_actions)

    # Checkpointing utils
    checkpointer = checkpoint.PyTreeCheckpointer()
    options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = checkpoint.CheckpointManager(
        os.path.abspath(config.logdir), checkpointer, options
    )

    # Train feature model and policy
    step = 0
    pbar = tqdm(total=config.training.num_steps)
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            rng, psi, policy, train_info = update(
                config,
                rng,
                psi,
                psi_sampler,
                psi_loss_fn,
                policy,
                policy_loss_fn,
                batch,
                num_actions,
            )
            wandb.log(train_info)

            # Evaluate
            if (step + 1) % config.training.eval_every == 0:
                rng, eval_info = evaluate(
                    config,
                    rng,
                    env,
                    planner,
                    psi,
                    psi_sampler,
                    policy,
                    policy_sampler,
                    w,
                    num_actions,
                )
                wandb.log(eval_info)

            # Save checkpoint
            if (step + 1) % config.training.save_every == 0:
                ckpt = {"config": config, "psi": psi, "policy": policy, "w": w}
                checkpoint_manager.save(step, ckpt)

            step += 1
            pbar.update(1)

        # Logging
        wandb.log({"train/epoch": epoch})

    pbar.close()


if __name__ == "__main__":
    train()