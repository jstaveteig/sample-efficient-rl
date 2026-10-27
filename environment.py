from functools import partial
from typing import Any

from flax import struct
import jax
import jax.numpy as jnp


class GymnaxWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)


class BatchEnvWrapper(GymnaxWrapper):
    def __init__(self, env, numEnvs):
        super().__init__(env)
        self.numEnvs = numEnvs
        self.resetFunction = jax.vmap(env.reset, in_axes=(0, None))
        self.stepFunction = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        unusedKey, resetKey = jax.random.split(rng)
        del unusedKey
        resetKeys = jax.random.split(resetKey, self.numEnvs)
        return self.resetFunction(resetKeys, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        unusedKey, stepKey = jax.random.split(rng)
        del unusedKey
        stepKeys = jax.random.split(stepKey, self.numEnvs)
        return self.stepFunction(stepKeys, state, action, params)


class AutoResetEnvWrapper(GymnaxWrapper):
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self.env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, stepKey = jax.random.split(rng)
        observations, nextState, reward, done, info = self.env.step(stepKey, state, action, params)
        unusedKey, resetKey = jax.random.split(rng)
        del unusedKey
        resetObservations, resetState = self.env.reset(resetKey, params)
        # retain terminal metrics while selecting the next episode
        nextState = jax.tree.map(
            lambda resetValue, stepValue: jax.lax.select(done, resetValue, stepValue),
            resetState,
            nextState,
        )
        observations = jax.lax.select(done, resetObservations, observations)
        return observations, nextState, reward, done, info


@struct.dataclass
class LogEnvState:
    envState: Any
    episodeReturns: float
    episodeLengths: int
    returnedEpisodeReturns: float
    returnedEpisodeLengths: int
    returnedEpisodeAchievements: jnp.ndarray
    timestep: int


class LogWrapper(GymnaxWrapper):
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        observations, envState = self.env.reset(key, params)
        state = LogEnvState(envState, 0.0, 0, 0.0, 0, jnp.zeros_like(envState.achievements), 0)
        return observations, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        observations, envState, reward, done, info = self.env.step(
            key, state.envState, action, params
        )
        episodeReturn = state.episodeReturns + reward
        episodeLength = state.episodeLengths + 1
        state = LogEnvState(
            envState=envState,
            episodeReturns=episodeReturn * (1 - done),
            episodeLengths=episodeLength * (1 - done),
            returnedEpisodeReturns=state.returnedEpisodeReturns * (1 - done) + episodeReturn * done,
            returnedEpisodeLengths=state.returnedEpisodeLengths * (1 - done) + episodeLength * done,
            returnedEpisodeAchievements=jnp.where(
                done,
                envState.achievements,
                state.returnedEpisodeAchievements,
            ),
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returnedEpisodeReturns
        info["returned_episode_lengths"] = state.returnedEpisodeLengths
        info["returned_episode_achievements"] = state.returnedEpisodeAchievements
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return observations, state, reward, done, info
