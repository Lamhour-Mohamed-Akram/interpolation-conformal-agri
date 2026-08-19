"""Conformal prediction primitives shared by all experiments.

Every experiment in this repository draws its split-conformal quantile and its
adaptive-conformal (ACI) recursion from this module, so a quantity that appears
in more than one experiment is computed by exactly one implementation.

Contents
--------
conformal_quantile   finite-sample conformal quantile of a score sample
scp_quantile         the same quantile applied to absolute residuals
scp_evaluate         coverage (PICP) and mean interval width (MPIW) of an SCP band
aci                  adaptive conformal inference with outcome-availability-aware
                     feedback

Finite-sample quantile
----------------------
The split-conformal quantile is the k-th order statistic of the calibration
scores with k = ceil((n + 1) * (1 - alpha)), clipped to [1, n]. It is computed
directly from the sorted sample rather than through NumPy's default quantile,
whose linear interpolation between order statistics is not the finite-sample
conformal rank.

Outcome-availability-aware feedback
-----------------------------------
The outcome of a forecast issued at origin bin o_j for target bin tau_j is
physically observable only from tau_j onward. In ``feedback_mode="observable"``
(the causally correct protocol) the recursion therefore holds each forecast's
(score, miss-indicator) pair in a pending queue keyed by its target bin, and
releases it into the online state before issuing a later forecast k if and
only if

    tau_j <= o_k,

i.e., the outcome time has been reached by the time forecast k is issued
(observability at equality is allowed: an outcome whose target bin coincides
with the next origin is treated as observed at that origin). This rule is
exact on filtered, gap-aware sequence streams, where the number of surviving
evaluation rows between two forecasts says nothing about the physical time
between them; a fixed row-count delay is NOT a valid substitute and is not
implemented here.

``feedback_mode="immediate"`` reproduces the deliberately defective
naive-pipeline variant, in which each forecast's outcome updates the state
from the very next forecast onward regardless of when it becomes observable.
It exists only as the methodological comparator studied in the paper.

Default parameters
------------------
ALPHA  0.10   nominal miss rate, i.e. 90% target coverage
GAMMA  0.02   ACI step size
WINDOW 500    residual pool length
"""
from __future__ import annotations

import numpy as np

ALPHA = 0.10
GAMMA = 0.02
WINDOW = 500


def conformal_quantile(scores, alpha: float = ALPHA) -> float:
    """Finite-sample conformal quantile of a score sample at level 1-alpha.

    Returns the k-th smallest score with k = ceil((n + 1) * (1 - alpha))
    clipped to [1, n]. NaN scores are dropped, which is what makes the
    gap-aware arm well defined: excluded bins simply do not contribute a
    calibration score.
    """
    s = np.asarray(scores, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(s)[k - 1])


def scp_quantile(residuals, alpha: float = ALPHA) -> float:
    """Split-conformal quantile of |residual| at level 1-alpha.

    The absolute residual is the conformity score; the quantile is the
    finite-sample order statistic from ``conformal_quantile``, so the resulting
    band carries the usual split-conformal marginal-coverage guarantee.
    """
    return conformal_quantile(np.abs(np.asarray(residuals, dtype=float)), alpha)


def scp_evaluate(res_cal, yhat, y, alpha: float = ALPHA) -> tuple[float, float]:
    """Coverage and mean width of the SCP band calibrated on ``res_cal``.

    Returns (PICP, MPIW) on the supplied evaluation set.
    """
    q = scp_quantile(res_cal, alpha)
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    picp = float(((y >= yhat - q) & (y <= yhat + q)).mean())
    return picp, float(2 * q)


def aci(res_cal, yhat, y, *, origin_bin=None, target_bin=None,
        feedback_mode: str = "observable", gamma: float = GAMMA,
        window: int = WINDOW, alpha: float = ALPHA, scale=None,
        return_traces: bool = False, return_diagnostics: bool = False):
    """Adaptive conformal inference with availability-aware feedback.

    Parameters
    ----------
    res_cal : calibration scores seeding the residual pool (absolute residuals,
              or sigma-normalised scores when ``scale`` is given). The caller
              is responsible for passing only scores whose outcomes are
              observable before the first evaluated forecast (the shared
              target-isolation rule guarantees this: every calibration target
              bin lies strictly before the first test origin bin).
    yhat, y : point forecasts and outcomes over the evaluation stream, in
              chronological origin order
    origin_bin, target_bin :
              per-forecast origin and outcome-availability bins on the
              physical grid (or timestamps in any common monotone unit).
              Required for ``feedback_mode="observable"``; ignored for
              ``feedback_mode="immediate"``.
    feedback_mode :
              "observable" — a forecast's (score, error) enters the online
              state before issuing forecast k iff its target bin satisfies
              target_bin[j] <= origin_bin[k] (the causally correct protocol);
              "immediate" — feedback enters at the next forecast regardless
              of observability (the deliberate naive-pipeline defect).
    scale   : optional per-step positive scale s_t; the interval is
              yhat_t +/- q * s_t and the online score is |y_t - yhat_t| / s_t
              (sigma-normalised ACI). Omitted: s_t = 1.

    Returns (coverage, mean interval width); with ``return_traces``,
    (lo, hi, covered) arrays; ``return_diagnostics`` appends a dict of causal
    feedback diagnostics (n_test, n_feedback_consumed,
    n_observable_before_last_origin, first_feedback_index,
    median_pending_at_issue, frac_issued_before_any_feedback).
    """
    if feedback_mode not in ("observable", "immediate"):
        raise ValueError(f"unknown feedback_mode {feedback_mode!r}")
    yhat = np.asarray(yhat, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    s = np.ones(n) if scale is None else np.asarray(scale, dtype=float)
    if feedback_mode == "observable":
        if origin_bin is None or target_bin is None:
            raise ValueError("feedback_mode='observable' requires origin_bin "
                             "and target_bin")
        origin_bin = np.asarray(origin_bin)
        target_bin = np.asarray(target_bin)
        if len(origin_bin) != n or len(target_bin) != n:
            raise ValueError("origin_bin/target_bin must match the stream length")
        if np.any(np.diff(origin_bin) < 0):
            raise ValueError("forecasts must be in chronological origin order")
        if np.any(target_bin <= origin_bin):
            raise ValueError("every target bin must lie after its origin bin")

    pool = list(np.abs(np.asarray(res_cal, dtype=float))[-window:])
    a_t = alpha
    cov = np.zeros(n, bool)
    lo = np.zeros(n, float)
    hi = np.zeros(n, float)
    pending: list[tuple[float, float, float]] = []   # (availability_bin, score, err)
    n_consumed = 0
    first_feedback_index = -1
    pending_sizes = np.zeros(n, dtype=np.int64)
    for t in range(n):
        if feedback_mode == "observable":
            now = origin_bin[t]
            due = [p for p in pending if p[0] <= now]
            pending = [p for p in pending if p[0] > now]
        else:
            due, pending = pending, []
        for avail, r, e in due:
            if feedback_mode == "observable":
                # no look-ahead, ever: the outcome must be observable now
                assert avail <= origin_bin[t]
            a_t += gamma * (alpha - e)
            pool.append(r)
            if len(pool) > window:
                pool.pop(0)
        if due and first_feedback_index < 0:
            first_feedback_index = t
        n_consumed += len(due)
        pending_sizes[t] = len(pending)
        eff = min(max(1 - a_t, 1e-3), 1 - 1e-3)
        q = float(np.quantile(pool, eff))
        lo[t], hi[t] = yhat[t] - q * s[t], yhat[t] + q * s[t]
        c = lo[t] <= y[t] <= hi[t]
        cov[t] = c
        avail_at = target_bin[t] if feedback_mode == "observable" else t
        pending.append((avail_at, abs(y[t] - yhat[t]) / s[t], float(not c)))

    out: tuple
    if return_traces:
        out = (lo, hi, cov)
    else:
        out = (float(cov.mean()), float((hi - lo).mean()))
    if return_diagnostics:
        if feedback_mode == "observable":
            n_obs_before_last = int((target_bin <= origin_bin[-1]).sum()) if n else 0
        else:
            n_obs_before_last = max(n - 1, 0)
        diag = dict(
            n_test=int(n),
            n_feedback_consumed=int(n_consumed),
            n_observable_before_last_origin=n_obs_before_last,
            first_feedback_index=int(first_feedback_index),
            median_pending_at_issue=float(np.median(pending_sizes)) if n else 0.0,
            frac_issued_before_any_feedback=(
                1.0 if first_feedback_index < 0 else first_feedback_index / n),
        )
        out = out + (diag,)
    return out
