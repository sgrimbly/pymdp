from typing import List

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.lax

from pymdp.agent import Agent
from pymdp.envs.env import Env


def default_policy_search(agent, qs, rng_key):
    qpi, G = agent.infer_policies(qs) # infer_policies computes posterior over policies using EFE
    return qpi, {"G": G}


def infer_and_plan(agent: Agent, qs_prev: List, observation: List, action_prev: jnp.array = None, rng_key: jr.PRNGKey = None, policy_search=None):
    """
    Perform a single inference and planning step for an agent in an environment.

    Parameters
    ----------
    agent: active inference agent
    qs_prev: previous posterior beliefs about hidden states
    observation: current observation from the environment
    action_prev: optional last action taken by the agent, if None agent.D is used as empirical prior
    rng_key: random key in cased the policy_search function needs to sample
    policy_search: optional custom policy inference function such as sophisticated inference
    """
    if policy_search is None:
        policy_search = default_policy_search

    if action_prev is not None:
        # updating the prior over hidden states and using B matrix to predict next state given action
        empirical_prior, _ = agent.update_empirical_prior(action_prev, qs_prev)
    else:
        # if no action is provided, use the agent's D as empirical prior
        empirical_prior = agent.D

    # infer states
    qs = agent.infer_states(
            observations=observation,
            empirical_prior=empirical_prior,
        )

    # get posterior over policies
    rng_key, key = jr.split(rng_key)
    qpi, xtra = policy_search(agent, qs, key) # compute policy posterior using EFE - uses C to consider preferred outcomes
    
    # for learning A and/or B
    if action_prev is not None:
        if agent.learn_A or agent.learn_B:
            if agent.learn_B:
                # stacking beliefs for B learning
                beliefs_B = jtu.tree_map(lambda x, y: jnp.concatenate([x,y], axis=1), qs_prev, qs)
                # reshaping action to match the stacked beliefs
                action_B = jnp.expand_dims(action_prev, 1)  # adding time dimension
            else:
                beliefs_B = None
                action_B = action_prev

            agent = agent.infer_parameters(
                qs, 
                observation, 
                action_B if agent.learn_B else action_prev,
                beliefs_B=beliefs_B
            )

    # sample action from policy distribution
    keys = jr.split(rng_key, agent.batch_size + 1)
    rng_key = keys[0]
    action = agent.sample_action(qpi, rng_key=keys[1:]) 

    info = {"empirical_prior": empirical_prior, "qpi": qpi}
    info.update(xtra)  # add extra information from policy search
    return action, qs, info


def rollout(agent: Agent, env: Env, num_timesteps: int, rng_key: jr.PRNGKey, initial_carry=None, policy_search=None):
    """
    Rollout an agent in an environment for a number of timesteps.

    Parameters
    ----------
    agent: active inference agent 
    env: environment that can step forward and return observations
    num_timesteps: how many timesteps to simulate
    rng_key: random key for sampling
    initial_carry: optional initial carry state to start the rollout from, if None it will be initialized from an environment reset
    policy_search: optional custom policy inference function such as sophisticated inference
    
    Returns
    ----------
    last: ``dict``
        dictionary from the last timestep about the rollout, i.e., the final action, observation, beliefs, etc.
    info: ``dict``
        dictionary containing information about the rollout, i.e. executed actions, observations, beliefs, etc.
    env: ``Env``
        environment state after the rollout
    """

    # get the batch_size of the agent
    batch_size = agent.batch_size

    # default policy search just uses standard active inference policy selection
    if policy_search is None:
        policy_search = default_policy_search

    def step_fn(carry, t):
        # carrying the current timestep's action, observation, beliefs, empirical prior, environment state, and random key
        action = carry["action"]
        observation = carry["observation"]
        qs_prev = carry["qs"]
        env = carry["env"]
        agent = carry["agent"]
        rng_key = carry["rng_key"]

        keys = jr.split(rng_key, batch_size + 2)
        rng_key = keys[0] # carry first key

        # infer next action and beliefs using the agent's inference and planning methods
        # first timestep we don't have a prev action yet, so then we call with action=None
        action_next, qs, xtra = jax.lax.cond(t > 0, lambda : infer_and_plan(agent, qs_prev, observation, action, keys[1], policy_search=policy_search), lambda : infer_and_plan(agent, qs_prev, observation, None, keys[1], policy_search=policy_search))
        
        # step the environment forward with the chosen action
        observation_next, env_next = env.step(rng_key=keys[2:], actions=action_next) # step environment forward with chosen action

        # carrying the next timestep's action, observation, beliefs, empirical prior, environment state, and random key
        carry = {
            "action": action_next,
            "observation": observation_next,
            "qs": jtu.tree_map(lambda x: x[:, -1:, ...], qs), # keep only latest belief
            "env": env_next,
            "agent": agent,
            "rng_key": rng_key,
        }
        info = {
            "qs": jtu.tree_map(lambda x: x[:, 0, ...], qs),
            "env": env,
            "observation": observation,
            "action": action_next,
        }
        info.update(xtra)  # add extra information from inference and planning

        return carry, info

    if initial_carry is None:
        # initialise first observation from environment
        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation_0, env = env.reset(keys[1:])

        # specify initial beliefs using D 
        qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)

        # get initial policy and action distribution - unused, just for shape matching
        qpi_0, _ = agent.infer_policies(qs_0)
        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        action_0 = agent.sample_action(qpi_0, rng_key=keys[1:])
        action_0 *= 0 # zero out initial action as no action taken yet

        # set up initial state to carry through timesteps
        initial_carry = {
            "qs": qs_0,
            "action": action_0,
            "observation": observation_0,
            "env": env,
            "agent": agent,
            "rng_key": rng_key,
        }

    # run the active inference loop for num_timesteps using jax.lax.scan
    last, info = jax.lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps + 1))

    info = jtu.tree_map(lambda x: x.transpose((1, 0) + tuple(range(2, x.ndim))), info)  # transpose to have timesteps as first dimension

    return last, info, env
