import functools
import os

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
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
from utils import get_planner
from user_int import get_user_input


def build_models(config, env, dataset, rng, num_actions):
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = np.prod(env.action_space.shape)
    feat_dim = env.feat_dim

    # Define learning rate schedule
    lr_fn = optax.warmup_cosine_decay_schedule(
        0, config.model.lr, config.model.warmup_steps, config.training.num_steps
    )

    # Initialize models
    psi_def = ConditionalUnet1D(
        output_dim=feat_dim,
        global_cond_dim=obs_dim,
        num_actions=num_actions,
        embed_dim=config.model.embed_dim,
        embed_type=config.model.embed_type,
    )
    rng, psi_rng = jax.random.split(rng)
    psi_params = psi_def.init(psi_rng, jnp.ones((1, feat_dim)), jnp.array([0.0]), jnp.ones((1, obs_dim)))["params"]
    psi = EMATrainState.create(
        model_def=psi_def,
        params=psi_params,
        ema_rate=config.model.ema_rate,
        tx=optax.adamw(learning_rate=lr_fn, weight_decay=config.model.wd),
    )
    psi_sde = VPSDE(
        beta_min=config.model.beta_min,
        beta_max=config.model.beta_max,
        N=config.model.num_steps,
    )
    psi_loss_fn = get_loss_fn(
        psi_sde,
        psi.model_def,
        lambda x: x,
        config.model.continuous,
    )
    psi_sampler = get_pc_sampler(
        psi_sde,
        psi.model_def,
        (feat_dim,),
        config.sampling.predictor,
        config.sampling.corrector,
        lambda x: x,
        config.model.continuous,
        config.sampling.n_inference_steps,
        eta=config.sampling.eta,
    )

    # Initialize policy model
    policy_def = ConditionalUnet1D(
        output_dim=act_dim * num_actions,
        global_cond_dim=obs_dim + feat_dim,
        num_actions=num_actions,
        embed_dim=config.model.embed_dim,
        embed_type=config.model.embed_type,
    )
    rng, policy_rng = jax.random.split(rng)
    policy_params = policy_def.init(
        policy_rng, jnp.ones((1, act_dim * num_actions)), jnp.array([0.0]), jnp.ones((1, obs_dim + feat_dim))
    )["params"]
    policy = EMATrainState.create(
        model_def=policy_def,
        params=policy_params,
        ema_rate=config.model.ema_rate,
        tx=optax.adamw(learning_rate=lr_fn, weight_decay=config.model.wd),
    )

    # Build planner
    planner = get_planner(
        config.planning.planner,
        lambda w: w.params["kernel"].T * config.planning.guidance_coef,
        config.planning.num_samples,
        config.planning.num_elites,
    )

    return psi, psi_sampler, psi_loss_fn, policy, planner, rng


@functools.partial(
    jax.jit,
    static_argnames=(
        "config",
        "psi_sampler",
        "psi_loss_fn",
        "policy_loss_fn",
        "num_actions",
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
    # Debugging: Print batch keys to verify correct format
    print("Batch keys:", batch.keys())

    # Ensure dataset keys exist
    obs_key = "observations" if "observations" in batch else "state"
    next_obs_key = "next_observations" if "next_observations" in batch else "next_state"

    if obs_key not in batch or next_obs_key not in batch:
        raise KeyError(f"Missing required keys in batch. Available keys: {batch.keys()}")

    observations = batch[obs_key]
    next_observations = batch[next_obs_key]

    # Sample target psi
    rng, sample_rng = jax.random.split(rng)
    next_psi = psi_sampler(psi.ema_params, sample_rng, next_observations)
    target_psi = batch["features"] + config.gamma * next_psi

    # Update psi model
    rng, loss_rng = jax.random.split(rng)
    psi, psi_info = psi.apply_loss_fn(
        loss_fn=psi_loss_fn,
        rng=loss_rng,
        x=target_psi,
        cond=observations,
        has_aux=True,
    )

    # Update policy
    rng, loss_rng = jax.random.split(rng)
    cond = jnp.concatenate([observations, target_psi], -1)
    actions = batch["actions"].reshape(-1, num_actions, observations.shape[-1])
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


@hydra.main(version_base=None, config_path="configs/", config_name="dispo.yaml")
def train(config):
    # Initialize wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "all-task-rl"),
        group=config.env_id,
        job_type=config.algo,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )

    # Get user input for num_actions
    num_actions = get_user_input()

    # Make environment and dataset
    env, dataset = make_env_and_dataset(
        config.env_id, config.seed, config.feat.type, config.feat.dim, num_actions
    )

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )

    # Define RNG
    rng = jax.random.PRNGKey(config.seed)

    # Build models
    psi, psi_sampler, psi_loss_fn, policy, planner, rng = build_models(config, env, dataset, rng, num_actions)

    # Train feature model and policy
    step = 0
    pbar = tqdm(total=config.training.num_steps)
    num_epochs = config.training.num_steps // len(dataloader)
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: jnp.array(v) for k, v in batch.items()}  # Convert batch to JAX-compatible arrays

            rng, psi, policy, train_info = update(
                config,
                rng,
                psi,
                psi_sampler,
                psi_loss_fn,
                policy,
                psi_loss_fn,  # Ensure only JAX-compatible arguments
                batch,
                num_actions,
            )
            wandb.log(train_info)

            step += 1
            pbar.update(1)

        wandb.log({"train/epoch": epoch})

    pbar.close()


if __name__ == "__main__":
    train()
