import functools
import jax
import jax.numpy as jnp
import optax


def batch_mul(a, b):
    """
    Perform element-wise multiplication between two arrays using `jax.vmap`.
    If the arrays have inconsistent batch sizes, the smaller array is truncated
    to match the size of the larger array.

    Args:
        a: First array (e.g., of shape [batch_size, ...]).
        b: Second array (e.g., of shape [batch_size, ...]).

    Returns:
        The result of element-wise multiplication between `a` and `b`.
    """
    # Ensure consistent batch sizes
    if a.shape[0] != b.shape[0]:
        min_size = min(a.shape[0], b.shape[0])  # Find the smaller batch size
        a = a[:min_size]  # Truncate a
        b = b[:min_size]  # Truncate b
    return jax.vmap(lambda a, b: a * b)(a, b)


def flatten_dict(config):
    """Flatten a hierarchical dict to a simple dict."""
    new_dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            sub_dict = flatten_dict(value)
            for subkey, subvalue in sub_dict.items():
                new_dict[key + "/" + subkey] = subvalue
        elif isinstance(value, tuple):
            new_dict[key] = str(value)
        else:
            new_dict[key] = value
    return new_dict


def clip_by_global_norm(max_norm):
    """Scale gradient updates using their global norm.

    Args:
      max_norm: The maximum global norm for an update.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        g_norm = optax.global_norm(updates)
        updates = jax.tree_util.tree_map(lambda t: (t / g_norm) * max_norm, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def get_planner(planner, guidance_fn, num_samples, num_elites):
    @functools.partial(
        jax.jit,
        static_argnames=("psi_sampler", "policy_sampler"),
    )
    def planner_fn(rng, psi, psi_sampler, policy, policy_sampler, w, obs):
        # Infer best psi
        rng, sample_rng = jax.random.split(rng)
        if planner == "random_shooting":
            obs_batch = obs.repeat(num_samples, 0)
            psis = psi_sampler(psi.ema_params, sample_rng, obs_batch)
            values = w(psis).sum(-1)
            sorted_inds = jnp.argsort(-values, axis=0)
            best_psi = psis[sorted_inds[:num_elites]].mean(axis=0, keepdims=True)
        elif planner == "guided_diffusion":
            g = guidance_fn(w)
            psis = psi_sampler(psi.ema_params, sample_rng, obs, g)
            best_psi = psis
        else:
            raise NotImplementedError(f"Unsupported planner: {planner}")

        # Predict action
        rng, sample_rng = jax.random.split(rng)
        action = policy_sampler(
            policy.ema_params, sample_rng, jnp.concatenate([obs, best_psi], -1)
        )

        # Store info
        info = {"psis": psis, "best_psi": best_psi}
        return rng, action, info

    return planner_fn