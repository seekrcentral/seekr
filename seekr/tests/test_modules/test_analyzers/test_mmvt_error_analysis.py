from __future__ import annotations
"""
test_modules/test_analyzers/test_mmvt_error_analysis.py

Validate the three MMVT error-analysis methods (bootstrap, resample,
montecarlo) against a simple analytic test system whose stationary
distribution pi_alpha is known in closed form.

The system is a linear chain of three anchors:

    anchor 0  <-->  anchor 1  <-->  anchor 2
       (endA)        (middle)        (endB)

with two internal milestones (global indices 0 and 1). Synthetic MMVT
anchor statistics are generated from a Poisson process with known
expected transition counts, so that the anchor-level detailed-balance
relation gives an exact analytic pi_alpha:

    pi[a] / pi[a-1] = N(a-1 -> a) / N(a -> a-1).

All three methods share the same point estimate (it comes from the main
data sample), so only the error bars differ between them. The tests are
organized in two tiers:

  * Tier A (fast, seeded): the point estimate matches the analytic
    distribution, and the reported pi_alpha error is finite, non-negative
    and non-degenerate.
  * Tier C (slow, @pytest.mark.slow): over many fresh realizations the
    mean reported error bar tracks the true sampling spread of the point
    estimate (statistical recovery).
"""

import random
from collections import defaultdict

import numpy as np
import pytest

import seekr.modules.structures as structures
import seekr.modules.analyzers.mmvt_anchor_statistics as mmvt_anchor_statistics
from seekr.modules.analyzers.mmvt_analyzer import Mmvt_analyzer

ALL_METHODS = ["bootstrap", "resample", "montecarlo"]

# Expected (Poisson-mean) transition counts over the full trajectory for
# each directed anchor pair in the 3-anchor chain. The ratios fix the
# analytic stationary distribution; the magnitudes set the sampling noise.
TRUE_COUNTS = {
    (0, 1): 300.0,   # anchor 0 -> anchor 1
    (1, 0): 200.0,   # anchor 1 -> anchor 0
    (1, 2): 225.0,   # anchor 1 -> anchor 2
    (2, 1): 150.0,   # anchor 2 -> anchor 1
}
# Fraction of the bounces off a milestone inside the middle anchor that
# are internal milestone-to-milestone transitions (in each direction).
# Drawing these as a binomial subset of the bounce counts keeps the
# statistics physically consistent: the number of transitions arriving at
# a milestone never exceeds the number of bounces off it, so the resample
# self-bounce diagonal (N_alpha_beta - incoming N_i_j) stays non-negative.
TRUE_INTERNAL_FRACTION = 0.5
# Total accrued simulation time per anchor (ps), used for T_alpha_total.
TOTAL_TIME = 1000.0
# ps per MD step; only sets step_count / time_count for the resample method.
STEP_DT = 0.002


def build_chain_model(
        num_anchors: int = 3,
        temperature: float = 298.15,
        ) -> structures.Seekr_model:
    """
    Build a minimal linear-chain Seekr_model with ``num_anchors`` anchors
    and ``num_anchors - 1`` internal milestones. Every non-bulk anchor is
    flagged with the "molecular_dynamics" scale type (required by the
    montecarlo method), and the two terminal anchors are marked as end
    anchors.
    """
    anchors = []
    for a in range(num_anchors):
        milestones = []
        neighbor_anchor_indices = []
        if a > 0:
            # Milestone toward the previous anchor: global index a - 1.
            milestones.append(structures.Milestone(
                index=a - 1,
                boundary=True,
                neighbor_anchor_index=a - 1,
                alias_index=1,
                variables={"avg_value": float(a - 1)}))
            neighbor_anchor_indices.append(a - 1)
        if a < num_anchors - 1:
            # Milestone toward the next anchor: global index a. The alias is
            # 1 for the leading end anchor (its only milestone) and 2 for an
            # interior anchor (its second milestone).
            milestones.append(structures.Milestone(
                index=a,
                boundary=True,
                neighbor_anchor_index=a + 1,
                alias_index=2 if a > 0 else 1,
                variables={"avg_value": float(a)}))
            neighbor_anchor_indices.append(a + 1)

        anchor = structures.Anchor(
            index=a,
            name="anchor_{}".format(a),
            directory="anchor_{}".format(a),
            bulk_anchor=False,
            scale_types=["molecular_dynamics"],
            end_anchor=(a == 0 or a == num_anchors - 1),
            neighbor_anchor_indices=neighbor_anchor_indices,
            milestones=milestones,
            variables={"value_0": float(a)})
        anchors.append(anchor)

    model = structures.Seekr_model(
        temperature=temperature,
        num_anchors=num_anchors,
        num_milestones=num_anchors - 1,
        anchors=anchors)
    return model


def analytic_pi_alpha(
        num_anchors: int = 3,
        true_counts: dict = TRUE_COUNTS,
        ) -> np.ndarray:
    """
    Return the normalized analytic stationary distribution over the
    non-bulk anchors, derived from anchor-level detailed balance:
    pi[a] / pi[a-1] = N(a-1 -> a) / N(a -> a-1).
    """
    pi = np.ones(num_anchors)
    for a in range(1, num_anchors):
        pi[a] = pi[a - 1] * true_counts[(a - 1, a)] / true_counts[(a, a - 1)]
    pi /= pi.sum()
    return pi


def make_synthetic_anchor_stats(
        model: structures.Seekr_model,
        rng: np.random.Generator,
        n_windows: int = 20,
        total_time: float = TOTAL_TIME,
        true_counts: dict = TRUE_COUNTS,
        internal_fraction: float = TRUE_INTERNAL_FRACTION,
        ) -> list[list[mmvt_anchor_statistics.Mmvt_anchor_statistics]]:
    """
    Generate cumulative MMVT anchor statistics windows for every anchor.

    For each anchor the counts accrue window by window as independent
    Poisson increments, so the final cumulative window holds the full
    statistics while the per-window differences (used by bootstrap) carry
    the correct Poisson variance. Counts are keyed by destination milestone
    alias, matching how the analysis pipeline reads real output files.
    """
    tau = total_time / n_windows
    anchor_stats_alpha_list = []
    for a, anchor in enumerate(model.anchors):
        # Directed edges out of this anchor: (dest alias, Poisson mean).
        edges = []
        for milestone in anchor.milestones:
            nb = milestone.neighbor_anchor_index
            edges.append((milestone.alias_index, true_counts.get((a, nb), 0.0)))
        interior = len(anchor.milestones) > 1

        cum_N = defaultdict(int)
        cum_N_i_j = defaultdict(int)
        cum_R = defaultdict(float)
        windows = []
        for w in range(n_windows):
            inc_N = {}
            for alias, mean_count in edges:
                inc = int(rng.poisson(mean_count / n_windows))
                inc_N[alias] = inc
                cum_N[alias] += inc
            if interior:
                # Internal milestone transitions (aliases 1 <-> 2) inside the
                # middle anchor are drawn as a binomial subset of the bounces
                # off the destination milestone, so the cumulative incoming
                # N_i_j never exceeds N_alpha_beta. Dwell time accrues at each
                # milestone.
                cum_N_i_j[(2, 1)] += int(
                    rng.binomial(inc_N[1], internal_fraction))
                cum_N_i_j[(1, 2)] += int(
                    rng.binomial(inc_N[2], internal_fraction))
                cum_R[1] += 0.5 * tau
                cum_R[2] += 0.5 * tau
            else:
                # Single-milestone end anchor: all dwell time at its milestone.
                only_alias = anchor.milestones[0].alias_index
                cum_R[only_alias] += tau

            stats = mmvt_anchor_statistics.Mmvt_anchor_statistics(alpha=a)
            stats.N_alpha_beta = defaultdict(int, dict(cum_N))
            stats.N_i_j_alpha = defaultdict(int, dict(cum_N_i_j))
            stats.R_i_alpha_total = defaultdict(float, dict(cum_R))
            stats.T_alpha_total = float((w + 1) * tau)
            stats.time_count = float((w + 1) * tau)
            stats.step_count = int(round((w + 1) * tau / STEP_DT))
            stats.compute_k_alpha_beta()
            windows.append(stats)

        anchor_stats_alpha_list.append(windows)
    return anchor_stats_alpha_list


def run_method(
        model: structures.Seekr_model,
        anchor_stats: list,
        method: str,
        num_error_samples: int,
        seed: int,
        time_between_statistics: float = TOTAL_TIME / 20,
        ) -> Mmvt_analyzer:
    """
    Drive a single error-analysis pass through the analyzer, bypassing all
    file I/O by injecting the synthetic anchor statistics directly. The
    montecarlo method additionally uses Python's stdlib ``random``, so it is
    seeded here as well as via ``error_analysis`` (which only seeds numpy).
    """
    analyzer = Mmvt_analyzer(model=model, stage_index=1)
    analyzer.anchor_stats_alpha_list = anchor_stats
    analyzer.fill_out_and_process_data_samples(time_between_statistics)
    if method == "montecarlo":
        random.seed(seed)
    analyzer.error_analysis(
        method, num_error_samples=num_error_samples, random_seed=seed)
    return analyzer


def point_estimate(analyzer: Mmvt_analyzer) -> np.ndarray:
    """Return the non-bulk anchor stationary distribution as a flat array."""
    pi_alpha = analyzer.main_data_window.pi_alpha
    return np.asarray(pi_alpha[:-1, 0], dtype=float)


# ---------------------------------------------------------------------------
# Tier A: fast, seeded sanity checks.
# ---------------------------------------------------------------------------

def test_point_estimate_matches_analytic():
    """
    The point estimate (shared by all methods) recovers the analytic
    stationary distribution of the 3-anchor chain. Averaged over several
    independent realizations the mean estimate is unbiased, so it is
    compared against the analytic distribution with a tight tolerance.
    """
    model = build_chain_model()
    analytic = analytic_pi_alpha()
    estimates = []
    for r in range(30):
        rng = np.random.default_rng(1234 + r)
        anchor_stats = make_synthetic_anchor_stats(model, rng)
        analyzer = run_method(
            model, anchor_stats, "bootstrap", num_error_samples=2, seed=r)
        estimates.append(point_estimate(analyzer))
    mean_estimate = np.mean(estimates, axis=0)
    assert np.allclose(mean_estimate, analytic, atol=0.02), \
        "mean pi_alpha {} not close to analytic {}".format(
            mean_estimate, analytic)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_pi_alpha_error_finite_and_positive(method):
    """
    Each method produces a finite, non-negative pi_alpha error bar, and a
    healthy fraction of the error samples yield a valid (non-None) pi_alpha.
    The valid-fraction guard catches the silent all-None degeneracy that
    would otherwise leave the error bars at an uninformative zero.
    """
    model = build_chain_model()
    rng = np.random.default_rng(2024)
    anchor_stats = make_synthetic_anchor_stats(model, rng)
    num_error_samples = 20 if method == "montecarlo" else 100
    analyzer = run_method(
        model, anchor_stats, method, num_error_samples=num_error_samples,
        seed=7)
    window = analyzer.main_data_window

    assert window.pi_alpha_err is not None
    err = np.asarray(window.pi_alpha_err[:-1, 0], dtype=float)
    assert np.all(np.isfinite(err)), "non-finite pi_alpha error: {}".format(err)
    assert np.all(err >= 0.0), "negative pi_alpha error: {}".format(err)
    assert np.any(err > 0.0), "all pi_alpha errors are zero (degenerate)"

    valid = sum(1 for ds in window.data_sample_list if ds.pi_alpha is not None)
    assert valid >= 0.5 * len(window.data_sample_list), \
        "only {}/{} error samples produced a valid pi_alpha".format(
            valid, len(window.data_sample_list))


# ---------------------------------------------------------------------------
# Tier C: slow statistical-recovery test.
# ---------------------------------------------------------------------------

# Per-method settings for the recovery test: number of fresh realizations
# and the number of error samples drawn within each realization. Montecarlo
# is markedly more expensive per sample, so it uses fewer of each.
_RECOVERY_SETTINGS = {
    "bootstrap":  {"num_realizations": 300, "num_error_samples": 120},
    "resample":   {"num_realizations": 300, "num_error_samples": 120},
    "montecarlo": {"num_realizations": 60, "num_error_samples": 30},
}


@pytest.mark.slow
@pytest.mark.parametrize("method", ALL_METHODS)
def test_pi_alpha_error_recovers_sampling_spread(method):
    """
    Statistical-recovery test: over many independent realizations of the
    synthetic data, the mean reported pi_alpha error bar should track the
    true standard deviation of the point estimate across realizations.

    For bootstrap and resample this is the frequentist sampling spread
    directly. For montecarlo the error bar is a Bayesian posterior
    (credible) interval, but under these well-sampled conditions the
    posterior SD coincides with the frequentist sampling SD, so the same
    comparison applies.
    """
    settings = _RECOVERY_SETTINGS[method]
    num_realizations = settings["num_realizations"]
    num_error_samples = settings["num_error_samples"]

    model = build_chain_model()
    n_windows = 10
    point_estimates = []
    reported_errors = []
    for r in range(num_realizations):
        rng = np.random.default_rng(10000 + r)
        anchor_stats = make_synthetic_anchor_stats(
            model, rng, n_windows=n_windows)
        analyzer = run_method(
            model, anchor_stats, method,
            num_error_samples=num_error_samples, seed=r)
        point_estimates.append(point_estimate(analyzer))
        reported_errors.append(
            np.asarray(analyzer.main_data_window.pi_alpha_err[:-1, 0],
                       dtype=float))

    point_estimates = np.array(point_estimates)
    reported_errors = np.array(reported_errors)

    sampling_std = point_estimates.std(axis=0, ddof=1)
    mean_reported_err = reported_errors.mean(axis=0)

    # Compare per anchor component; the reported error should land within a
    # factor of ~2 of the empirical sampling spread (the error-analysis
    # methods are known to mildly under- or over-estimate).
    for component in range(len(sampling_std)):
        if sampling_std[component] < 1e-6:
            continue
        ratio = mean_reported_err[component] / sampling_std[component]
        assert 0.5 < ratio < 2.0, \
            "{}: component {} reported err {:.4g} vs sampling std {:.4g} " \
            "(ratio {:.2f})".format(
                method, component, mean_reported_err[component],
                sampling_std[component], ratio)
