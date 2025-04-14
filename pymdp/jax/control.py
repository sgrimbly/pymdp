#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

import itertools
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import List, Tuple, Optional
from functools import partial, reduce
from jax.scipy.special import xlogy
from jax import lax, jit, vmap, nn
from jax import random as jr
from itertools import chain
from jaxtyping import Array

from pymdp.jax.maths import *
# import pymdp.jax.utils as utils

def get_marginals(q_pi, policies, num_controls):
    """
    Computes the marginal posterior(s) over actions by integrating their posterior probability under the policies that they appear within.

    Parameters
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    num_controls: ``list`` of ``int``
        ``list`` of the dimensionalities of each control state factor.
    
    Returns
    ----------
    action_marginals: ``list`` of ``jax.numpy.ndarrays``
       List of arrays corresponding to marginal probability of each action possible action
    """
    num_factors = len(num_controls)    

    action_marginals = []
    for factor_i in range(num_factors):
        actions = jnp.arange(num_controls[factor_i])[:, None]
        action_marginals.append(jnp.where(actions==policies[:, 0, factor_i], q_pi, 0).sum(-1))
    
    return action_marginals

def sample_action(policies, num_controls, q_pi, action_selection="deterministic", alpha=16.0, rng_key=None):
    """
    Samples an action from posterior marginals, one action per control factor.

    Parameters
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    num_controls: ``list`` of ``int``
        ``list`` of the dimensionalities of each control state factor.
    action_selection: string, default "deterministic"
        String indicating whether whether the selected action is chosen as the maximum of the posterior over actions,
        or whether it's sampled from the posterior marginal over actions
    alpha: float, default 16.0
        Action selection precision -- the inverse temperature of the softmax that is used to scale the 
        action marginals before sampling. This is only used if ``action_selection`` argument is "stochastic"

    Returns
    ----------
    selected_policy: 1D ``numpy.ndarray``
        Vector containing the indices of the actions for each control factor
    """

    marginal = get_marginals(q_pi, policies, num_controls)
    
    if action_selection == 'deterministic':
        selected_policy = jtu.tree_map(lambda x: jnp.argmax(x, -1), marginal)
    elif action_selection == 'stochastic':
        logits = lambda x: alpha * log_stable(x)
        selected_policy = jtu.tree_map(lambda x: jr.categorical(rng_key, logits(x)), marginal)
    else:
        raise NotImplementedError

    return jnp.array(selected_policy)

def sample_policy(policies, q_pi, action_selection="deterministic", alpha = 16.0, rng_key=None):

    if action_selection == "deterministic":
        policy_idx = jnp.argmax(q_pi)
    elif action_selection == "stochastic":
        log_p_policies = log_stable(q_pi)
        scaled_logits = log_p_policies * jnp.array(alpha).reshape(1)
        policy_idx = jr.categorical(key=rng_key, logits=scaled_logits)
        # print("Selected policy indices:", policy_idx)
        # print("Selected policies:", policies[policy_idx])
        
    selected_multiaction = policies[policy_idx, 0]
    return selected_multiaction

# def sample_policy(policies, q_pi, action_selection="deterministic", alpha=1.0, rng_key=None):
#     """
#     Sample policies using either deterministic or stochastic selection
    
#     Args:
#         policies (Array): Policy array of shape (num_policies, num_timesteps, action_dim)
#         q_pi (Array): Policy probabilities of shape (batch_size, num_policies)
#         action_selection (str): Selection method ('deterministic' or 'stochastic')
#         alpha (float): Temperature parameter for softmax
#         rng_key (Array): Random key of shape (batch_size, 2)
    
#     Returns:
#         Array: Selected actions of shape (batch_size, action_dim)
#     """
#     print(f"Input shapes: q_pi {q_pi.shape}, rng_key {rng_key.shape}")
#     if action_selection == "deterministic":
#         policy_idx = jnp.argmax(q_pi, axis=-1)
#     elif action_selection == "stochastic":
#         log_p_policies = log_stable(q_pi) * alpha
#         policy_idx = jr.categorical(key=rng_key, logits=log_p_policies, shape=(q_pi.shape[0],))
                
#     # Get first action from selected policies
#     batch_idx = jnp.arange(policy_idx.shape[0])
#     selected_multiaction = policies[policy_idx][batch_idx, 0]
    
#     return selected_multiaction

def construct_policies(num_states, num_controls = None, policy_len=1, control_fac_idx=None):
    """
    Generate a ``list`` of policies. The returned array ``policies`` is a ``list`` that stores one policy per entry.
    A particular policy (``policies[i]``) has shape ``(num_timesteps, num_factors)`` 
    where ``num_timesteps`` is the temporal depth of the policy and ``num_factors`` is the number of control factors.

    Parameters
    ----------
    num_states: ``list`` of ``int``
        ``list`` of the dimensionalities of each hidden state factor
    num_controls: ``list`` of ``int``, default ``None``
        ``list`` of the dimensionalities of each control state factor. If ``None``, then is automatically computed as the dimensionality of each hidden state factor that is controllable
    policy_len: ``int``, default 1
        temporal depth ("planning horizon") of policies
    control_fac_idx: ``list`` of ``int``
        ``list`` of indices of the hidden state factors that are controllable (i.e. those state factors ``i`` where ``num_controls[i] > 1``)

    Returns
    ----------
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    """

    num_factors = len(num_states)
    if control_fac_idx is None:
        if num_controls is not None:
            control_fac_idx = [f for f, n_c in enumerate(num_controls) if n_c > 1]
        else:
            control_fac_idx = list(range(num_factors))

    if num_controls is None:
        num_controls = [num_states[c_idx] if c_idx in control_fac_idx else 1 for c_idx in range(num_factors)]
        
    x = num_controls * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))
    
    for pol_i in range(len(policies)):
        policies[pol_i] = jnp.array(policies[pol_i]).reshape(policy_len, num_factors)

    return jnp.stack(policies)


def update_posterior_policies(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies, B_dependencies, gamma=16.0, use_utility=True, use_states_info_gain=True, use_param_info_gain=False):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies,
                                     use_utility=use_utility, use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain)

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies

# def compute_expected_state(qs_prior, B, u_t, B_dependencies=None): 
#     """
#     Compute posterior over next state, given belief about previous state, transition model and action...
#     """
#     #Note: this algorithm is only correct if each factor depends only on itself. For any interactions, 
#     # we will have empirical priors with codependent factors. 
#     assert len(u_t) == len(B)  
#     qs_next = []
#     for B_f, u_f, deps in zip(B, u_t, B_dependencies):
#         relevant_factors = [qs_prior[idx] for idx in deps]
#         qs_next_f = factor_dot(B_f[...,u_f], relevant_factors, keep_dims=(0,))
#         qs_next.append(qs_next_f)
        
#     # P(s'|s, u) = \sum_{s, u} P(s'|s) P(s|u) P(u|pi)P(pi) because u </-> pi
#     return qs_next

# def compute_expected_state_and_Bs(qs_prior, B, u_t): 
#     """
#     Compute posterior over next state, given belief about previous state, transition model and action...
#     """
#     assert len(u_t) == len(B)  
#     qs_next = []
#     Bs = []
#     for qs_f, B_f, u_f in zip(qs_prior, B, u_t):
#         qs_next.append( B_f[..., u_f].dot(qs_f) )
#         Bs.append(B_f[..., u_f])
    
#     return qs_next, Bs

# def compute_expected_obs(qs, A, A_dependencies):
#     """
#     New version of expected observation (computation of Q(o|pi)) that takes into account sparse dependencies between observation
#     modalities and hidden state factors
#     """
        
#     def compute_expected_obs_modality(A_m, m):
#         deps = A_dependencies[m]
#         relevant_factors = [qs[idx] for idx in deps]
#         return factor_dot(A_m, relevant_factors, keep_dims=(0,))

#     return jtu.tree_map(compute_expected_obs_modality, A, list(range(len(A))))

# def compute_info_gain(qs, qo, A, A_dependencies):
#     """
#     New version of expected information gain that takes into account sparse dependencies between observation modalities and hidden state factors.
#     """

#     def compute_info_gain_for_modality(qo_m, A_m, m):
#         H_qo = stable_entropy(qo_m)
#         H_A_m = - stable_xlogx(A_m).sum(0)
#         deps = A_dependencies[m]
#         relevant_factors = [qs[idx] for idx in deps]
#         qs_H_A_m = factor_dot(H_A_m, relevant_factors)
#         return H_qo - qs_H_A_m
    
#     info_gains_per_modality = jtu.tree_map(compute_info_gain_for_modality, qo, A, list(range(len(A))))
        
#     return jtu.tree_reduce(lambda x,y: x+y, info_gains_per_modality)

# def compute_expected_utility(t, qo, C):
    
#     util = 0.
#     for o_m, C_m in zip(qo, C):
#         if C_m.ndim > 1:
#             util += (o_m * C_m[t]).sum()
#         else:
#             util += (o_m * C_m).sum()
    
#     return util

# def calc_pA_info_gain(pA, qo, qs, A_dependencies):
#     """
#     Compute expected Dirichlet information gain about parameters ``pA`` for a given posterior predictive distribution over observations ``qo`` and states ``qs``.

#     Parameters
#     ----------
#     pA: ``numpy.ndarray`` of dtype object
#         Dirichlet parameters over observation model (same shape as ``A``)
#     qo: ``list`` of ``numpy.ndarray`` of dtype object
#         Predictive posterior beliefs over observations; stores the beliefs about
#         observations expected under the policy at some arbitrary time ``t``
#     qs: ``list`` of ``numpy.ndarray`` of dtype object
#         Predictive posterior beliefs over hidden states, stores the beliefs about
#         hidden states expected under the policy at some arbitrary time ``t``

#     Returns
#     -------
#     infogain_pA: float
#         Surprise (about Dirichlet parameters) expected for the pair of posterior predictive distributions ``qo`` and ``qs``
#     """

#     def infogain_per_modality(pa_m, qo_m, m):
#         wa_m = spm_wnorm(pa_m) * (pa_m > 0.)
#         fd = factor_dot(wa_m, [s for f, s in enumerate(qs) if f in A_dependencies[m]], keep_dims=(0,))[..., None]
#         return qo_m.dot(fd)

#     pA_infogain_per_modality = jtu.tree_map(
#         infogain_per_modality, pA, qo, list(range(len(qo)))
#     )
    
#     infogain_pA = jtu.tree_reduce(lambda x, y: x + y, pA_infogain_per_modality)
#     return infogain_pA.squeeze(-1)

# def calc_pB_info_gain(pB, qs_t, qs_t_minus_1, B_dependencies, u_t_minus_1):
#     """
#     Compute expected Dirichlet information gain about parameters ``pB`` under a given policy

#     Parameters
#     ----------
#     pB: ``Array`` of dtype object
#         Dirichlet parameters over transition model (same shape as ``B``)
#     qs_t: ``list`` of ``Array`` of dtype object
#         Predictive posterior beliefs over hidden states expected under the policy at time ``t``
#     qs_t_minus_1: ``list`` of ``Array`` of dtype object
#         Posterior over hidden states at time ``t-1`` (before receiving observations)
#     u_t_minus_1: "Array"
#         Actions in time step t-1 for each factor

#     Returns
#     -------
#     infogain_pB: float
#         Surprise (about Dirichlet parameters) expected under the policy in question
#     """
    
#     wB = lambda pb:  spm_wnorm(pb) * (pb > 0.)
#     fd = lambda x, i: factor_dot(x, [s for f, s in enumerate(qs_t_minus_1) if f in B_dependencies[i]], keep_dims=(0,))[..., None]
    
#     pB_infogain_per_factor = jtu.tree_map(lambda pb, qs, f: qs.dot(fd(wB(pb[..., u_t_minus_1[f]]), f)), pB, qs_t, list(range(len(qs_t))))
#     infogain_pB = jtu.tree_reduce(lambda x, y: x + y, pB_infogain_per_factor)[0]
#     return infogain_pB

# --- Helper Function (Ensure this is available) ---
def _has_leaf_dim(state_tensor, param_tensor):
     """ Checks if state tensor likely has an extra leading dim compared to param tensor """
     if param_tensor is None or not hasattr(param_tensor, 'ndim') or param_tensor.ndim == 0:
          return False
     if not hasattr(state_tensor, 'ndim'):
         return False # Cannot compare if state_tensor doesn't have ndim
     # Simple heuristic: state has more dims than param suggests L dim
     try:
         # Base param ndim expects batch + other + state dims
         # Base state ndim expects batch + state dims
         # So if state_tensor.ndim > param_tensor.ndim - 1 (account for 'other' like obs dim), L might exist
         # This is still fragile. Using the simple ndim check for now.
         return state_tensor.ndim > param_tensor.ndim
     except:
         return False # Fallback

def batched_multidimensional_outer(list_of_arrays):
    """
    Calculates the outer product along the last axes of arrays in a list,
    preserving leading batch dimensions (L?, b).
    """
    if not list_of_arrays:
        return jnp.array(1.0)
    if len(list_of_arrays) == 1:
        return list_of_arrays[0]
    return reduce(lambda x, y: jnp.einsum('...i,...j->...ij', x, y), list_of_arrays)

# --- MODIFIED compute_expected_state ---
def compute_expected_state(qs_prior, B, u_t, B_dependencies=None):
    """
    Compute P(s'|pi), potentially handling Leaf dim L via explicit vmap.
    Uses explicit vmap over agent batch `b` for core contraction.
    Replaces *all* inner einsum calls with matmul (@) or reshape+matmul.
    qs_prior shapes [(L?, b, s_f)], B shapes [(b, s_f', s_deps..., u)]
    """
    assert len(u_t) == len(B)
    qs_next = []

    param_for_dim_check = B[0] if len(B) > 0 else None
    # Check if qs_prior[0] exists and has dimensions before checking ndim
    has_L_dim = False
    if len(qs_prior) > 0 and hasattr(qs_prior[0], 'ndim'):
        if param_for_dim_check is not None and hasattr(param_for_dim_check, 'ndim') and param_for_dim_check.ndim > 0:
             # Simple heuristic: state has more dims than param suggests L dim
             # This check needs careful validation based on actual param shapes
             try: # Be robust if param_tensor doesn't have expected dims
                 has_L_dim = qs_prior[0].ndim > param_for_dim_check.ndim
             except:
                 has_L_dim = False # Fallback
        elif param_for_dim_check is None or not hasattr(param_for_dim_check, 'ndim') or param_for_dim_check.ndim == 0:
             has_L_dim = qs_prior[0].ndim > 2


    for f, (B_f, u_f, deps) in enumerate(zip(B, u_t, B_dependencies)):
        B_f_u = B_f[..., u_f] # Shape (b, s_f_next, s_deps...)

        if not deps: # Single factor dependency (self dependency, e.g. deps=[0])
             # This block assumes deps is like [0], not []
             if not isinstance(deps, list) or len(deps) != 1:
                  # If deps is truly empty, handle in the `else` block below
                  # This path is only for single self-dependency
                   raise ValueError(f"Expected deps=[f] for self-dependency, got: {deps}")


             qs_f = qs_prior[f] # Shape (L?, b, s)

             # Define inner operation for a single batch element `b`
             # Input B_slice shape (S, s), Input q_slice shape (s,) -> Output shape (S,)
             # Use matmul: B_slice @ q_slice
             inner_op = lambda B_slice, q_slice: B_slice @ q_slice

             # Vmap the inner op over the agent batch `b` dimension (axis=0)
             vmapped_op_b = vmap(inner_op, in_axes=(0, 0))

             if has_L_dim:
                 qs_next_f = vmap(vmapped_op_b, in_axes=(None, 0))(B_f_u, qs_f) # Output (L, b, S)
             else:
                 qs_next_f = vmapped_op_b(B_f_u, qs_f) # Output (b, S)

        elif len(deps) > 0: # Multi-factor dependency (e.g. deps = [0, 1])
             relevant_factors = [qs_prior[idx] for idx in deps] # Shapes [(L?, b, s_dep_i)]
             qs_joint_deps = batched_multidimensional_outer(relevant_factors) # Shape (L?, b, s...)

             # Define inner operation for a single batch element `b` using reshape + matmul
             # Input B_slice shape (S, s0, s1...), Input q_slice shape (s0, s1...) -> Output shape (S,)
             def inner_op_matmul(B_slice, q_slice):
                  # Flatten state dimensions
                  S_dim = B_slice.shape[0]
                  B_flat = B_slice.reshape(S_dim, -1) # Shape (S, s0*s1*...)
                  q_flat = q_slice.reshape(-1)       # Shape (s0*s1*...,)
                  return B_flat @ q_flat             # Output shape (S,)

             # Vmap the inner op over the agent batch `b` dimension (axis=0)
             vmapped_op_b = vmap(inner_op_matmul, in_axes=(0, 0))

             if has_L_dim:
                 qs_next_f = vmap(vmapped_op_b, in_axes=(None, 0))(B_f_u, qs_joint_deps) # Output (L, b, S)
             else:
                 qs_next_f = vmapped_op_b(B_f_u, qs_joint_deps) # Output (b, S)
        else: # Case where deps = [] -> Factor evolves independently of any state
             # B_f_u should have shape (b, S)
             if B_f_u.ndim != 2:
                  raise ValueError(f"B matrix for independent factor {f} has wrong shape {B_f_u.shape}, expected (b, S)")
             qs_next_f = B_f_u # Shape (b, S)
             if has_L_dim:
                  # Find L dim size from another factor's qs_prior if possible
                  L_dim = 1
                  if len(qs_prior)>0 and hasattr(qs_prior[0], 'ndim') and qs_prior[0].ndim > B_f_u.ndim:
                      L_dim = qs_prior[0].shape[0]
                  elif len(qs_prior) > 1 and hasattr(qs_prior[1], 'ndim') and qs_prior[1].ndim > B_f_u.ndim: # Check next factor
                      L_dim = qs_prior[1].shape[0]
                  else:
                      print(f"Warning: Could not determine L dimension for broadcasting independent factor {f}")

                  if L_dim > 1:
                      qs_next_f = jnp.broadcast_to(qs_next_f, (L_dim,) + qs_next_f.shape) # Shape (L, b, S)


        qs_next.append(qs_next_f)

    return qs_next


# --- MODIFIED compute_expected_obs ---
def compute_expected_obs(qs_pi, A, A_dependencies):
    """
    Calculates P(o|pi), potentially handling Leaf dim L via explicit vmap.
    Uses explicit vmap over agent batch `b` for core contraction.
    qs_pi shapes [(L?, b, s_f)], A shapes [(b, o, s_deps...)]
    """
    qo_pi = []
    num_modalities = len(A)

    param_for_dim_check = A[0] if len(A) > 0 else None
    has_L_dim = _has_leaf_dim(qs_pi[0], param_for_dim_check)

    for m in range(num_modalities):
        A_m = A[m]
        dep_indices = A_dependencies[m]

        if not dep_indices:
            # State-independent modality: Need P(o) broadcasted to (L?, b, o)
            # ... (previous broadcasting logic - seems okay) ...
            n_obs_m = A_m.shape[-1] if hasattr(A_m, 'ndim') and A_m.ndim > 0 else 1
            target_batch_shape = qs_pi[0].shape[:-1] # (L?, b)
            target_shape = target_batch_shape + (n_obs_m,) if hasattr(A_m, 'ndim') and A_m.ndim > 0 else target_batch_shape
            temp_A_m = A_m
            missing_batch_dims = len(target_batch_shape) - (A_m.ndim - 1 if hasattr(A_m, 'ndim') and A_m.ndim > 0 else 0)
            if missing_batch_dims > 0 and hasattr(A_m, 'ndim') and A_m.ndim > 0 :
                 temp_A_m = jnp.expand_dims(A_m, axis=tuple(range(missing_batch_dims)))
            elif missing_batch_dims > 0 and (not hasattr(A_m, 'ndim') or A_m.ndim == 0):
                 temp_A_m = A_m
            try:
                if not hasattr(A_m, 'ndim') or A_m.ndim == 0:
                     qo_m = jnp.broadcast_to(temp_A_m, target_batch_shape)
                     if n_obs_m == 1: qo_m = jnp.expand_dims(qo_m, axis=-1)
                else:
                     qo_m = jnp.broadcast_to(temp_A_m, target_shape)
            except ValueError as e:
                 print(f"Warning: Broadcasting state-independent modality {m} failed. A_m shape: {A_m.shape}, Target shape: {target_shape}. Error: {e}")
                 qo_m = temp_A_m
            qo_pi.append(qo_m)
            continue

        # State-dependent modality
        qs_factors_m = [qs_pi[f] for f in dep_indices]
        qs_outer = batched_multidimensional_outer(qs_factors_m) # Shape (L?, b, s...)

        num_state_factors = len(dep_indices)
        state_subscripts = "abcdefghijklmnopqrstuvwxyz"[:num_state_factors] # s...
        obs_subscript = "o"

        # Define inner operation for a single batch element `b`
        # Input A_slice shape (o, s...), Input q_slice shape (s...)
        # Output shape (o,)
        einsum_str_single_b = f'{obs_subscript}{state_subscripts},{state_subscripts}->{obs_subscript}'
        inner_op = lambda A_slice, q_slice: jnp.einsum(einsum_str_single_b, A_slice, q_slice)

        # Vmap the inner op over the agent batch `b` dimension (axis=0)
        # Inputs A_m shape (b, o, s...), qs_outer shape (b, s...) -> needs adjustment if L present
        # Output shape (b, o)
        vmapped_op_b = vmap(inner_op, in_axes=(0, 0))

        if has_L_dim:
            # Also vmap over the L dimension (axis=0 of qs_outer)
            # A_m is fixed w.r.t L dim
            qo_m = vmap(vmapped_op_b, in_axes=(None, 0))(A_m, qs_outer) # Output (L, b, o)
        else:
            qo_m = vmapped_op_b(A_m, qs_outer) # Output (b, o)

        qo_pi.append(qo_m)

    return qo_pi


# --- MODIFIED compute_info_gain ---
def compute_info_gain(qs, qo, A, A_dependencies):
    """
    Compute expected information gain EIG = H[P(o|pi)] - E_qs[H[P(o|s)]].
    Handles potential Leaf dim L via explicit vmap over agent batch `b` and L dim.
    Sums over L and b dims at the end to return scalar EFE component.
    """
    # Check leaf dim presence (heuristic using first state factor and first A matrix)
    param_for_dim_check = A[0] if len(A) > 0 else None
    has_L_dim = False
    if len(qs) > 0 and hasattr(qs[0], 'ndim'):
        if param_for_dim_check is not None and hasattr(param_for_dim_check, 'ndim') and param_for_dim_check.ndim > 0:
             try: has_L_dim = qs[0].ndim > param_for_dim_check.ndim
             except: has_L_dim = False # Fallback
        elif param_for_dim_check is None or not hasattr(param_for_dim_check, 'ndim') or param_for_dim_check.ndim == 0:
             # If no param or param is scalar, assume L dim if state has > 2 dims (b + s)
             has_L_dim = qs[0].ndim > 2

    def compute_info_gain_for_modality(qo_m, A_m, m):
        # qo_m shape (L?, b, o)

        # --- Calculate H[P(o|pi)] = H_qo ---
        # Calculate entropy per L?, b element by vmapping stable_entropy
        # Assuming stable_entropy works on 1D array (o,) -> scalar output
        entropy_1d = stable_entropy # Assign the imported function
        vmapped_entropy_b = vmap(entropy_1d, in_axes=0) # Maps over b: Input (b, o) -> Output (b,)
        if has_L_dim:
            vmapped_entropy_Lb = vmap(vmapped_entropy_b, in_axes=0) # Maps over L: Input (L, b, o) -> Output (L, b)
            H_qo_Lb = vmapped_entropy_Lb(qo_m)
        else:
            H_qo_Lb = vmapped_entropy_b(qo_m) # Output (b,)

        # --- Calculate E_qs[H[P(o|s)]] = expected_H_A ---
        # Check if A_m is valid for calculating conditional entropy
        if not hasattr(A_m, 'ndim') or A_m.ndim <= 1:
            # If A_m is scalar or just (o,), P(o|s) = P(o). E[H(o|s)] = H(o) = H_qo. Info gain is zero.
            expected_H_A = H_qo_Lb # Set expected_H_A equal to H_qo to yield zero IG
        else:
            # H_A_m_orig = - sum_o A(o|s) log A(o|s) -- Calculated per state s
            obs_axis = 1 # Axis of observation dimension in A_m (b, o, s...)
            H_A_m_orig = - stable_xlogx(A_m).sum(axis=obs_axis) # Shape (b, s...) or (b,)

            deps = A_dependencies[m]
            if not deps: # State-independent (A_m was (b, o) or similar)
                expected_H_A = H_A_m_orig # Shape (b,)
                # Broadcast to match H_qo_Lb shape (L?, b) if L is present
                if has_L_dim and expected_H_A.ndim < H_qo_Lb.ndim:
                     expected_H_A = jnp.expand_dims(expected_H_A, axis=0) # Add L dim
                # Ensure shape matches H_qo via broadcasting (handles L=1 vs L>1)
                if expected_H_A.shape != H_qo_Lb.shape:
                     expected_H_A = jnp.broadcast_to(expected_H_A, H_qo_Lb.shape)
            else: # State-dependent calculation
                 relevant_factors = [qs[idx] for idx in deps] # Shapes [(L?, b, s_i)]
                 qs_joint_deps = batched_multidimensional_outer(relevant_factors) # Shape (L?, b, s...)

                 num_state_factors = len(deps)
                 state_subscripts = "abcdefghijklmnopqrstuvwxyz"[:num_state_factors] # s...

                 # Inner op for single batch element `b`: E_q(s)[ H(o|s) ] = sum_s q(s)H(o|s)
                 # Input H_slice shape (s...), Input q_slice shape (s...) -> Output scalar
                 einsum_str_single_b = f'{state_subscripts},{state_subscripts}->'
                 inner_op = lambda H_slice, q_slice: jnp.einsum(einsum_str_single_b, H_slice, q_slice)

                 # Vmap over b dim
                 vmapped_op_b = vmap(inner_op, in_axes=(0, 0)) # Input (b, s...), (b, s...) -> Output (b,)

                 if has_L_dim:
                     # Vmap over L dim of qs_joint_deps, H_A_m_orig is fixed w.r.t L
                     expected_H_A = vmap(vmapped_op_b, in_axes=(None, 0))(H_A_m_orig, qs_joint_deps) # Output (L, b)
                 else:
                     expected_H_A = vmapped_op_b(H_A_m_orig, qs_joint_deps) # Output (b,)

        # --- Calculate Info Gain = H_qo - expected_H_A ---
        # Ensure shapes match before subtraction
        if expected_H_A.shape != H_qo_Lb.shape:
            try:
                # Attempt broadcasting one last time
                expected_H_A = jnp.broadcast_to(expected_H_A, H_qo_Lb.shape)
                print(f"Note: Broadcasted expected_H_A from {expected_H_A.shape} to {H_qo_Lb.shape} in modality {m}")
            except ValueError:
                # If broadcasting fails, shapes are fundamentally incompatible. Log error, return 0.
                print(f"ERROR: Could not broadcast expected_H_A ({expected_H_A.shape}) to H_qo_Lb ({H_qo_Lb.shape}) in modality {m}. Returning zero IG for this modality.")
                return jnp.zeros_like(H_qo_Lb) # Return zeros of the correct batch shape

        return H_qo_Lb - expected_H_A

    # Calculate info gain per modality (each resulting in shape (L?, b))
    info_gains_per_modality = jtu.tree_map(compute_info_gain_for_modality, qo, A, list(range(len(A))))

    # Sum info gains across modalities --> shape (L?, b)
    total_info_gain_Lb = jtu.tree_reduce(lambda x,y: x+y, info_gains_per_modality)

    # Return sum over L?, b dimensions -> scalar
    return total_info_gain_Lb.sum()


# --- MODIFIED compute_expected_utility ---
def compute_expected_utility(t, qo, C):
    """
    Computes expected utility E[U] = sum_{o} P(o|pi) * C(o).
    Handles qo with shape (L?, b, o) and C with various shapes (b, T?, o), (b, o), (o,), scalar.
    Uses einsum directly for batch dimension b, and vmap for leaf dimension L if present.
    """
    util = 0.
    # Check leaf dim presence from first qo modality
    has_L_dim = qo[0].ndim > 2 # Assumes qo[f] is at least (b, o)

    for m, (o_m, C_m) in enumerate(zip(qo, C)):
        # o_m shape (L?, b, o)

        # Determine C_m_t for the current timestep t
        C_m_t = C_m
        C_has_b_dim = False # Flag if C has agent batch dimension
        if hasattr(C_m, 'ndim'):
             if C_m.ndim == 3: # Shape (b, T, o)
                 C_m_t = C_m[:, t, :] # Shape (b, o)
                 C_has_b_dim = True
             elif C_m.ndim == 2: # Shape (b, o)
                 C_m_t = C_m # Shape (b, o)
                 C_has_b_dim = True
             elif C_m.ndim == 1: # Shape (o,)
                 C_m_t = C_m
             elif C_m.ndim == 0: # Scalar
                 C_m_t = C_m
             else:
                 raise ValueError(f"Utility C_m shape {C_m.shape} unsupported")
        else: # Scalar
            C_m_t = C_m

        # Define the utility calculation for a single L-slice (operating on b dim)
        # Input: o_m_slice shape (b, o), C_m_t shape (b, o) or (o,) or scalar
        # Output: util_slice shape (b,)
        def utility_op_for_b_dim(o_m_slice, C_m_t_arg):
            if hasattr(C_m_t_arg, 'ndim') and C_m_t_arg.ndim >= 2 and C_has_b_dim: # C is (b, o)
                # einsum: o_m(b,o), C_m_t(b,o) -> util(b,)
                return jnp.einsum('bo,bo->b', o_m_slice, C_m_t_arg)
            elif hasattr(C_m_t_arg, 'ndim') and C_m_t_arg.ndim == 1: # C is (o,)
                # einsum: o_m(b,o), C_m_t(o,) -> util(b,)
                return jnp.einsum('bo,o->b', o_m_slice, C_m_t_arg)
            else: # C is scalar
                return o_m_slice.sum(axis=-1) * C_m_t_arg # Shape (b,)

        # Apply the operation over L?, b dimensions
        if has_L_dim:
             # Vmap the b-dimension operation over the L dimension (axis 0 of o_m)
             # C_m_t needs to be handled correctly based on whether it has b dim
             if C_has_b_dim:
                 # Map over o_m (L, b, o) and C_m_t (b, o) - C_m_t is fixed for L vmap
                 util_m_Lb = vmap(utility_op_for_b_dim, in_axes=(0, None))(o_m, C_m_t) # Output (L, b)
             else:
                 # Map over o_m (L, b, o), C_m_t ((o,) or scalar) is fixed
                 util_m_Lb = vmap(utility_op_for_b_dim, in_axes=(0, None))(o_m, C_m_t) # Output (L, b)
        else: # No L dim
             # Apply directly to the b dimension
             util_m_Lb = utility_op_for_b_dim(o_m, C_m_t) # Output (b,)

        util += util_m_Lb.sum() # Sum over L?, b dims

    return util


# --- MODIFIED calc_pA_info_gain ---
def calc_pA_info_gain(pA, qo, qs, A_dependencies):
    """ Compute EIG about pA, using explicit vmap over agent batch `b`. """
    param_for_dim_check = pA[0] if len(pA) > 0 else None
    has_L_dim = _has_leaf_dim(qs[0], param_for_dim_check)

    def infogain_per_modality(pa_m, qo_m, m):
        # pa_m shape (b, o, s...) | qo_m shape (L?, b, o)
        if not hasattr(pa_m, 'ndim') or pa_m.ndim <= 1: return 0.0

        wa_m = spm_wnorm(pa_m) * (pa_m > 0.) # Shape (b, o, s...)

        deps = A_dependencies[m]
        if not deps: return 0.0

        relevant_factors = [qs[idx] for idx in deps] # Shapes [(L?, b, s_i)]
        qs_joint_deps = batched_multidimensional_outer(relevant_factors) # Shape (L?, b, s...)

        num_state_factors = len(deps)
        state_subscripts = "abcdefghijklmnopqrstuvwxyz"[:num_state_factors] # s...
        obs_subscript = "o"

        # Calculate fd = E_qs[wnorm(pA)] per batch element b
        # Inner op for single b: inputs wa_slice(o,s...), q_slice(s...) -> output (o,)
        einsum_str_fd_single_b = f'{obs_subscript}{state_subscripts},{state_subscripts}->{obs_subscript}'
        inner_op_fd = lambda wa_slice, q_slice: jnp.einsum(einsum_str_fd_single_b, wa_slice, q_slice)
        vmapped_op_fd_b = vmap(inner_op_fd, in_axes=(0, 0)) # Map over b

        # Apply over L?, b
        if has_L_dim:
             fd = vmap(vmapped_op_fd_b, in_axes=(None, 0))(wa_m, qs_joint_deps) # Output (L, b, o)
        else:
             fd = vmapped_op_fd_b(wa_m, qs_joint_deps) # Output (b, o)

        # Final contraction: E_qo[ fd ]
        # Inner op for single b: inputs qo_slice(o,), fd_slice(o,) -> output scalar
        einsum_str_final_single_b = f'{obs_subscript},{obs_subscript}->'
        inner_op_final = lambda qo_slice, fd_slice: jnp.einsum(einsum_str_final_single_b, qo_slice, fd_slice)
        vmapped_op_final_b = vmap(inner_op_final, in_axes=(0, 0)) # Map over b

        # Apply over L?, b
        if has_L_dim:
             infogain_m_Lb = vmap(vmapped_op_final_b, in_axes=(0, 0))(qo_m, fd) # Output (L, b)
        else:
             infogain_m_Lb = vmapped_op_final_b(qo_m, fd) # Output (b,)

        return infogain_m_Lb.sum() # Sum over L?, b

    pA_infogain_per_modality = jtu.tree_map(infogain_per_modality, pA, qo, list(range(len(qo))))
    infogain_pA = jtu.tree_reduce(lambda x, y: x + y, pA_infogain_per_modality)
    return infogain_pA


# --- MODIFIED calc_pB_info_gain ---
def calc_pB_info_gain(pB, qs_t, qs_t_minus_1, B_dependencies, u_t_minus_1):
    """ Compute EIG about pB, using explicit vmap over agent batch `b`. """
    param_for_dim_check = pB[0] if len(pB) > 0 else None
    has_L_dim = _has_leaf_dim(qs_t[0], param_for_dim_check)

    def wB_fn(pb):
        # Add check for zero variance/division by zero if needed by spm_wnorm
        return spm_wnorm(pb) * (pb > 0.)

    def infogain_per_factor(pb_f, qs_f_t, f):
        # pb_f shape (b, S, s..., u) | qs_f_t shape (L?, b, S)
        if not hasattr(pb_f, 'ndim') or pb_f.ndim <= 2: return 0.0

        u_f = u_t_minus_1[f]
        deps = B_dependencies[f]

        pb_f_u = pb_f[..., u_f] # Shape (b, S, s...)
        wB_f_u = wB_fn(pb_f_u) # Shape (b, S, s...)

        num_state_deps = len(deps) if deps else 1
        state_subscripts = "abcdefghijklmnopqrstuvwxyz"[:num_state_deps] # s...
        next_state_subscript = "S"

        # Calculate fd = E_qs_tm1[wnorm(pB)] per batch element b
        # Inner op for single b: inputs wB_slice(S,s...), q_slice(s...) -> output (S,)
        einsum_str_fd_single_b = f'{next_state_subscript}{state_subscripts},{state_subscripts}->{next_state_subscript}'
        inner_op_fd = lambda wB_slice, q_slice: jnp.einsum(einsum_str_fd_single_b, wB_slice, q_slice)
        vmapped_op_fd_b = vmap(inner_op_fd, in_axes=(0, 0)) # Map over b

        # Apply over L?, b
        if not deps: # Factor depends only on itself
            qs_f_tm1 = qs_t_minus_1[f] # Shape (L?, b, s)
            if has_L_dim:
                fd = vmap(vmapped_op_fd_b, in_axes=(None, 0))(wB_f_u, qs_f_tm1) # Output (L, b, S)
            else:
                fd = vmapped_op_fd_b(wB_f_u, qs_f_tm1) # Output (b, S)
        else: # Factor depends on multiple factors
            relevant_factors_tm1 = [qs_t_minus_1[idx] for idx in deps]
            qs_joint_deps_tm1 = batched_multidimensional_outer(relevant_factors_tm1) # Shape (L?, b, s...)
            if has_L_dim:
                fd = vmap(vmapped_op_fd_b, in_axes=(None, 0))(wB_f_u, qs_joint_deps_tm1) # Output (L, b, S)
            else:
                fd = vmapped_op_fd_b(wB_f_u, qs_joint_deps_tm1) # Output (b, S)

        # Final contraction: E_qs_t[ fd ]
        # Inner op for single b: inputs q_t_slice(S,), fd_slice(S,) -> output scalar
        einsum_str_final_single_b = f'{next_state_subscript},{next_state_subscript}->'
        inner_op_final = lambda q_t_slice, fd_slice: jnp.einsum(einsum_str_final_single_b, q_t_slice, fd_slice)
        vmapped_op_final_b = vmap(inner_op_final, in_axes=(0, 0)) # Map over b

        # Apply over L?, b
        if has_L_dim:
            infogain_f_Lb = vmap(vmapped_op_final_b, in_axes=(0, 0))(qs_f_t, fd) # Output (L, b)
        else:
            infogain_f_Lb = vmapped_op_final_b(qs_f_t, fd) # Output (b,)

        return infogain_f_Lb.sum() # Sum over L?, b

    pB_infogain_per_factor = jtu.tree_map(infogain_per_factor, pB, qs_t, list(range(len(qs_t))))
    infogain_pB = jtu.tree_reduce(lambda x, y: x + y, pB_infogain_per_factor)
    return infogain_pB




def compute_G_policy(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, policy_i, use_utility=True, use_states_info_gain=True, use_param_info_gain=False):
    """ Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop. """

    def scan_body(carry, t):

        qs, neg_G = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain = compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.

        utility = compute_expected_utility(qo, C) if use_utility else 0.

        param_info_gain = calc_pA_info_gain(pA, qo, qs_next) if use_param_info_gain else 0.
        param_info_gain += calc_pB_info_gain(pB, qs_next, qs, policy_i[t]) if use_param_info_gain else 0.

        neg_G += info_gain + utility + param_info_gain

        return (qs_next, neg_G), None

    qs = qs_init
    neg_G = 0.
    final_state, _ = lax.scan(scan_body, (qs, neg_G), jnp.arange(policy_i.shape[0]))
    qs_final, neg_G = final_state
    return neg_G

def compute_G_policy_inductive(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, policy_i, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=False):
    """ 
    Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop.
    This one further adds computations used for inductive planning.
    """

    def scan_body(carry, t):

        qs, neg_G = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain = compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.

        utility = compute_expected_utility(t, qo, C) if use_utility else 0.

        inductive_value = calc_inductive_value_t(qs_init, qs_next, I, epsilon=inductive_epsilon) if use_inductive else 0.

        param_info_gain = 0.
        if pA is not None:
            param_info_gain += calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        if pB is not None:
            param_info_gain += calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.

        neg_G += info_gain + utility - param_info_gain + inductive_value

        return (qs_next, neg_G), None

    qs = qs_init
    neg_G = 0.
    final_state, _ = lax.scan(scan_body, (qs, neg_G), jnp.arange(policy_i.shape[0]))
    _, neg_G = final_state
    return neg_G

def update_posterior_policies_inductive(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies, B_dependencies, I, gamma=16.0, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=True):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy_inductive, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, inductive_epsilon=inductive_epsilon,
                                     use_utility=use_utility,  use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain, use_inductive=use_inductive)

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies

def generate_I_matrix(H: List[Array], B: List[Array], threshold: float, depth: int):
    """ 
    Generates the `I` matrices used in inductive planning. These matrices stores the probability of reaching the goal state backwards from state j (columns) after i (rows) steps.
    Parameters
    ----------    
    H: ``list`` of ``jax.numpy.ndarray``
        Constraints over desired states (1 if you want to reach that state, 0 otherwise)
    B: ``list`` of ``jax.numpy.ndarray``
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    threshold: ``float``
        The threshold for pruning transitions that are below a certain probability
    depth: ``int``
        The temporal depth of the backward induction

    Returns
    ----------
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.
    """
    
    num_factors = len(H)
    I = []
    for f in range(num_factors):
        """
        For each factor, we need to compute the probability of reaching the goal state
        """

        # If there exists an action that allows transitioning 
        # from state to next_state, with probability larger than threshold
        # set b_reachable[current_state, previous_state] to 1
        b_reachable = jnp.where(B[f] > threshold, 1.0, 0.0).sum(axis=-1)
        b_reachable = jnp.where(b_reachable > 0., 1.0, 0.0)

        def step_fn(carry, i):
            I_prev = carry
            I_next = jnp.dot(b_reachable, I_prev)
            I_next = jnp.where(I_next > 0.1, 1.0, 0.0) # clamp I_next to 1.0 if it's above 0.1, 0 otherwise
            return I_next, I_next
    
        _, I_f = lax.scan(step_fn, H[f], jnp.arange(depth-1))
        I_f = jnp.concatenate([H[f][None,...], I_f], axis=0)

        I.append(I_f)
    
    return I

def calc_inductive_value_t(qs, qs_next, I, epsilon=1e-3):
    """
    Computes the inductive value of a state at a particular time (translation of @tverbele's `numpy` implementation of inductive planning, formerly
    called `calc_inductive_cost`).

    Parameters
    ----------
    qs: ``list`` of ``jax.numpy.ndarray`` 
        Marginal posterior beliefs over hidden states at a given timepoint.
    qs_next: ```list`` of ``jax.numpy.ndarray`` 
        Predictive posterior beliefs over hidden states expected under the policy.
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.
    epsilon: ``float``
        Value that tunes the strength of the inductive value (how much it contributes to the expected free energy of policies)

    Returns
    -------
    inductive_val: float
        Value (negative inductive cost) of visiting this state using backwards induction under the policy in question
    """
    
    # initialise inductive value
    inductive_val = 0.

    log_eps = log_stable(epsilon)
    for f in range(len(qs)):
        # we also assume precise beliefs here?!
        idx = jnp.argmax(qs[f])
        # m = arg max_n p_n < sup p

        # i.e. find first entry at which I_idx equals 1, and then m is the index before that
        m = jnp.maximum(jnp.argmax(I[f][:, idx])-1, 0)
        I_m = (1. - I[f][m, :]) * log_eps
        path_available = jnp.clip(I[f][:, idx].sum(0), min=0, max=1) # if there are any 1's at all in that column of I, then this == 1, otherwise 0
        inductive_val += path_available * I_m.dot(qs_next[f]) # scaling by path_available will nullify the addition of inductive value in the case we find no path to goal (i.e. when no goal specified)

    return inductive_val

# if __name__ == '__main__':

#     from jax import random as jr
#     key = jr.PRNGKey(1)
#     num_obs = [3, 4]

#     A = [jr.uniform(key, shape = (no, 2, 2)) for no in num_obs]
#     B = [jr.uniform(key, shape = (2, 2, 2)), jr.uniform(key, shape = (2, 2, 2))]
#     C = [log_stable(jnp.array([0.8, 0.1, 0.1])), log_stable(jnp.ones(4)/4)]
#     policy_1 = jnp.array([[0, 1],
#                          [1, 1]])
#     policy_2 = jnp.array([[1, 0],
#                          [0, 0]])
#     policy_matrix = jnp.stack([policy_1, policy_2]) # 2 x 2 x 2 tensor
    
#     qs_init = [jnp.ones(2)/2, jnp.ones(2)/2]
#     neg_G_all_policies = jit(update_posterior_policies)(policy_matrix, qs_init, A, B, C)
#     print(neg_G_all_policies)
