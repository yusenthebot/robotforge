"""A training run in a few lines: what an agent reads from training_status (the viewer draws the same data).

The status file holds every trial's curves for the viewer; an answer to an agent keeps the trials side by side with
their settings, where each is, a handful of points of its curve and its latest losses, and says which leads.
"""

from __future__ import annotations

from typing import Any

CURVE_POINTS = 6


def _few(series: list[list[float]], n: int = CURVE_POINTS) -> list[list[float]]:
    if len(series) <= n:
        return series
    step = (len(series) - 1) / (n - 1)
    return [series[round(i * step)] for i in range(n)]


def trial_view(t: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": t["id"],
        "algo": t["algo"],
        "settings": t.get("settings", {}),
        "state": t.get("state"),
        "steps": t.get("steps", 0),
    }
    if t.get("mean_return") is not None:
        out["recent_return"] = t["mean_return"]
    if t.get("success_rate") is not None:
        out["recent_success"] = t["success_rate"]
    score = t.get("score")
    if score:
        out["test_return"] = score.get("mean_return")
        out["test_success"] = score.get("success_rate")
        if score.get("mean_result"):
            out["test_result"] = score["mean_result"]
    if t.get("curve"):
        out["curve"] = [[int(x), y] for x, y in _few(t["curve"])]
    losses = {k: v[-1][1] for k, v in (t.get("metrics") or {}).items() if v}
    if losses:
        out["latest"] = losses
    if t.get("error"):
        out["error"] = t["error"]
    if t.get("timed_out"):
        out["timed_out"] = True
    return out


def run_view(status: dict[str, Any]) -> dict[str, Any]:
    """The trials of a run with the leader marked, or {} for a run without trials (an older status file)."""
    trials = status.get("trials")
    if not trials:
        return {}
    out: dict[str, Any] = {
        "leader": status.get("leader"),
        "trials": [trial_view(t) for t in trials],
    }
    if status.get("search", {}).get("method"):
        out["search"] = status["search"]
    if status.get("notes"):
        out["notes"] = status["notes"]
    if status.get("timed_out"):
        out["timed_out"] = True
    final = status.get("final")
    if final:
        out["winner"] = winner_view(final)
    out["read_me"] = (
        "Each trial is one algorithm with one set of settings. test_return and test_success come from the same "
        "fixed test starts for every trial, so only these compare trials fairly; recent_return averages the "
        "latest training episodes (ARS and CEM try perturbed policies there, so those sit lower and noisier than "
        "the policy's own test return) and does not rank anything. In a halving search, state 'pruned' trials "
        "were dropped for trailing and the best go on with more steps. The winner's test numbers are optimistic "
        "(it was picked for them); its held_out numbers are scored again on starts no trial was ranked on and are "
        "the ones to quote. The viewer's Training panel draws the curves and plays the best policy."
    )
    return out


def winner_view(final: dict[str, Any]) -> dict[str, Any]:
    """The winning trial: its settings, its score on the ranking starts and its score on held-out starts."""
    out: dict[str, Any] = {
        "trial": final.get("trial"),
        "algo": final.get("algo"),
        "settings": final.get("settings", {}),
        "ranking_return": final.get("mean_return"),
        "ranking_success": final.get("success_rate"),
    }
    held = final.get("held_out")
    if final.get("score_note"):
        out["score_note"] = final["score_note"]
    if held:
        out["held_out"] = {
            "episodes": held.get("episodes"),
            "return": held.get("mean_return"),
            "success": held.get("success_rate"),
            "result": held.get("mean_result", {}),
        }
    return out
