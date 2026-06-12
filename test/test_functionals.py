#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for the planning-functional axis (efe | feef | fef) in pymdp.control.

The functionals are related by exact identities (Millidge, Tschantz & Buckley 2021),
which the tests check against independent manual rollouts:

    neg_efe  = utility + info_gain (+ param_info_gain)
    neg_fef  = utility - info_gain (- param_info_gain)      => neg_efe + neg_fef = 2 * utility
    neg_feef = utility + ambiguity + info_gain               => neg_feef - neg_efe = ambiguity
             = utility + H[Q(o|pi)]                          (since info_gain = H[Q(o)] - ambiguity)
"""

import unittest

import numpy as np
import jax.numpy as jnp
import jax.random as jr

import pymdp.control as ctl
from pymdp.agent import Agent
from pymdp.maths import stable_entropy
from pymdp.utils import random_A_array, random_B_array, random_factorized_categorical


def _small_model(key, policy_len=2):
    num_states = [3, 2]
    num_obs = [4, 3]
    num_controls = [2, 1]
    A_dependencies = [[0, 1], [1]]
    B_dependencies = [[0], [1]]
    key_A, key_B, key_qs, key_C = jr.split(key, 4)
    A = random_A_array(key_A, num_obs, num_states, A_dependencies=A_dependencies)
    B = random_B_array(key_B, num_states, num_controls, B_dependencies=B_dependencies)
    qs = random_factorized_categorical(key_qs, num_states)
    # non-trivial preferences so the utility term is exercised
    C = [jr.uniform(k, (no,), minval=-2.0, maxval=2.0)
         for no, k in zip(num_obs, jr.split(key_C, len(num_obs)))]
    policy_matrix = ctl.construct_policies(num_states, num_controls, policy_len=policy_len)
    E = jnp.ones(policy_matrix.shape[0]) / policy_matrix.shape[0]
    return (num_states, num_obs, num_controls, A_dependencies, B_dependencies,
            A, B, C, qs, policy_matrix, E)


class TestPlanningFunctionals(unittest.TestCase):

    def _score(self, model, functional, **kwargs):
        (_, _, _, A_deps, B_deps, A, B, C, qs, policy_matrix, E) = model
        defaults = dict(use_utility=True, use_states_info_gain=True, use_param_info_gain=False)
        defaults.update(kwargs)
        _, neg_score = ctl.update_posterior_policies(
            policy_matrix, qs, A, B, C, E, None, None, A_deps, B_deps,
            functional=functional, **defaults)
        return neg_score

    def test_default_is_efe(self):
        model = _small_model(jr.PRNGKey(11))
        (_, _, _, A_deps, B_deps, A, B, C, qs, policy_matrix, E) = model
        _, neg_default = ctl.update_posterior_policies(
            policy_matrix, qs, A, B, C, E, None, None, A_deps, B_deps)
        neg_efe = self._score(model, "efe")
        self.assertTrue(np.allclose(np.array(neg_default), np.array(neg_efe)))

    def test_fef_is_resigned_info_gain(self):
        """neg_efe + neg_fef == 2 * utility-only score (state info gain cancels)."""
        model = _small_model(jr.PRNGKey(12))
        neg_efe = self._score(model, "efe")
        neg_fef = self._score(model, "fef")
        utility_only = self._score(model, "efe", use_states_info_gain=False)
        self.assertTrue(np.allclose(np.array(neg_efe + neg_fef),
                                    np.array(2 * utility_only), atol=1e-5))
        # and the FEF strictly discourages information gain wherever the EFE rewards it
        self.assertFalse(np.allclose(np.array(neg_efe), np.array(neg_fef)))

    def test_fef_resigns_param_info_gain_too(self):
        """With pB provided, the parameter info-gain term is also re-signed under fef."""
        model = _small_model(jr.PRNGKey(13))
        (_, _, _, A_deps, B_deps, A, B, C, qs, policy_matrix, E) = model
        key_pB = jr.PRNGKey(14)
        pB = [jr.uniform(k, b_f.shape, minval=0.5, maxval=2.0)
              for b_f, k in zip(B, jr.split(key_pB, len(B)))]

        def score(functional):
            _, neg = ctl.update_posterior_policies(
                policy_matrix, qs, A, B, C, E, None, pB, A_deps, B_deps,
                use_utility=True, use_states_info_gain=True, use_param_info_gain=True,
                functional=functional)
            return neg

        utility_only = self._score(model, "efe", use_states_info_gain=False)
        self.assertTrue(np.allclose(np.array(score("efe") + score("fef")),
                                    np.array(2 * utility_only), atol=1e-5))

    def test_feef_identities_via_manual_rollout(self):
        """neg_feef - neg_efe == accumulated ambiguity, and
        neg_feef == accumulated (utility + H[Q(o|pi)]) -- both from an independent rollout."""
        model = _small_model(jr.PRNGKey(15))
        (_, _, _, A_deps, B_deps, A, B, C, qs, policy_matrix, E) = model
        neg_efe = np.array(self._score(model, "efe"))
        neg_feef = np.array(self._score(model, "feef"))

        n_policies, policy_len, _ = policy_matrix.shape
        amb_sum = np.zeros(n_policies)
        util_plus_hqo = np.zeros(n_policies)
        for p in range(n_policies):
            qs_t = qs
            for t in range(policy_len):
                qs_t = ctl.compute_expected_state(qs_t, B, policy_matrix[p, t], B_deps)
                qo_t = ctl.compute_expected_obs(qs_t, A, A_deps)
                amb_sum[p] += float(ctl.compute_expected_ambiguity(qs_t, A, A_deps))
                util_plus_hqo[p] += float(ctl.compute_expected_utility(qo_t, C, t))
                util_plus_hqo[p] += sum(float(stable_entropy(qo_m)) for qo_m in qo_t)

        self.assertTrue(np.allclose(neg_feef - neg_efe, amb_sum, atol=1e-4))
        self.assertTrue(np.allclose(neg_feef, util_plus_hqo, atol=1e-4))

    def test_unknown_functional_raises(self):
        model = _small_model(jr.PRNGKey(16))
        with self.assertRaises(ValueError):
            self._score(model, "geff")

    def test_agent_level_functionals(self):
        """Agent(functional=...) threads through infer_policies; identities hold batched."""
        num_obs = [3, 2]
        num_states = [3, 2]
        num_controls = [2, 1]
        A_deps = [[0, 1], [1]]
        B_deps = [[0], [1]]
        key_A, key_B, key_C = jr.split(jr.PRNGKey(17), 3)
        A = random_A_array(key_A, num_obs, num_states, A_dependencies=A_deps)
        B = random_B_array(key_B, num_states, num_controls, B_dependencies=B_deps)
        C = [jr.uniform(k, (no,), minval=-2.0, maxval=2.0)
             for no, k in zip(num_obs, jr.split(key_C, len(num_obs)))]

        def make(functional, **kw):
            flags = dict(use_utility=True, use_states_info_gain=True)
            flags.update(kw)
            return Agent(A=A, B=B, C=C, A_dependencies=A_deps, B_dependencies=B_deps,
                         num_controls=num_controls, policy_len=2,
                         functional=functional, **flags)

        obs = [jnp.array([[1]]), jnp.array([[0]])]
        scores = {}
        for functional in ("efe", "feef", "fef"):
            agent = make(functional)
            qs = agent.infer_states(obs, empirical_prior=agent.D)
            _, scores[functional] = agent.infer_policies(qs)
        agent_u = make("efe", use_states_info_gain=False)
        qs = agent_u.infer_states(obs, empirical_prior=agent_u.D)
        _, utility_only = agent_u.infer_policies(qs)

        self.assertTrue(np.allclose(np.array(scores["efe"] + scores["fef"]),
                                    np.array(2 * utility_only), atol=1e-5))
        self.assertTrue(np.all(np.array(scores["feef"] - scores["efe"]) > -1e-5))

        with self.assertRaises(ValueError):
            make("not-a-functional")


if __name__ == "__main__":
    unittest.main()
