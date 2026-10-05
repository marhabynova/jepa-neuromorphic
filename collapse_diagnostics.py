import torch


@torch.no_grad()
def effective_rank(embeddings, normalize_rows=False):
    """Two DISTINCT definitions exist in this project's own code, confirmed by
    Opus audit round 2 (2026-10-02) to NOT be interchangeable -- a prior note
    in RESEARCH_NOTES.md §21g wrongly claimed they were "PERSIS SAMA" as
    EventCLR's version and used that to claim cross-validation; that claim is
    RETRACTED (§24 in the notes).

    - erank_S   (this project's original, Roy & Vetterli): entropy of the
      distribution p_i = s_i / sum(s_j)  -- linear in singular values.
    - erank_S2  (EventCLR's `utils/metrics.py`): entropy of
      p_i = s_i^2 / sum(s_j^2) -- entropy of the eigenvalue/variance spectrum,
      systematically LOWER than erank_S for the same spectrum.

    EventCLR also row-normalizes (projects onto the unit sphere) before SVD;
    this function additionally exposes that as `normalize_rows` so both of
    EventCLR's preprocessing choices (squared values + row-normalization) can
    be reproduced explicitly, not assumed equivalent to ours.

    Report BOTH `effective_rank_S` and `effective_rank_S2` when comparing
    across repos -- do not average, substitute, or treat one as validating
    the other."""
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    if normalize_rows:
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    _, s, _ = torch.linalg.svd(embeddings, full_matrices=False)

    # BUG FOUND 2026-10-02 (while verifying Opus audit round 2, random-init
    # test on EventCLR): clamp(min=1e-12) on ALL-ZERO singular values (true
    # total collapse -- embeddings identically zero, e.g. LIF neurons that
    # haven't fired at all at init) makes every clamped value equal, which
    # gives a UNIFORM distribution -> MAXIMUM entropy -> reports "full rank,
    # no collapse" for the worst possible collapse case. Guard explicitly.
    if s.sum() < 1e-10:
        return {
            "effective_rank_S": 0.0,
            "effective_rank_S2": 0.0,
            "max_possible_rank": min(embeddings.shape),
            "top1_singular_value_variance_fraction": float("nan"),
            "singular_values": [0.0] * min(10, s.shape[0]),
            "degenerate_all_zero": True,
        }
    s = s.clamp(min=1e-12)

    p_s = s / s.sum()
    erank_s = (-(p_s * p_s.log()).sum()).exp().item()

    p_s2 = (s ** 2) / (s ** 2).sum()
    erank_s2 = (-(p_s2 * p_s2.log()).sum()).exp().item()

    top1_variance_frac = (s[0] ** 2 / (s ** 2).sum()).item()
    return {
        "effective_rank_S": erank_s,
        "effective_rank_S2": erank_s2,
        "max_possible_rank": min(embeddings.shape),
        "top1_singular_value_variance_fraction": top1_variance_frac,
        "singular_values": s[:10].tolist(),
        "degenerate_all_zero": False,
    }


@torch.no_grad()
def per_dimension_variance(embeddings, relative_eps=1e-3):
    """relative_eps is a FRACTION of mean variance, not an absolute threshold --
    an absolute cutoff (e.g. var < 1e-6) is scale-dependent: it misclassifies
    collapse on embeddings with naturally small norms (e.g. rate-coded spike
    outputs averaged over a short window) independent of whether they're
    actually degenerate. Caught by Opus audit round 2, 2026-10-02."""
    var = embeddings.var(dim=0)
    mean_var = var.mean()
    threshold = relative_eps * mean_var
    return {
        "mean_variance": mean_var.item(),
        "min_variance": var.min().item(),
        "max_variance": var.max().item(),
        "fraction_near_zero_variance": (var < threshold).float().mean().item(),
    }


def diagnose(embeddings, label=""):
    """Run all collapse diagnostics and print a clear verdict."""
    rank_info = effective_rank(embeddings)
    var_info = per_dimension_variance(embeddings)

    print(f"\n=== Collapse diagnostics: {label} ===")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(f"Effective rank (S, ours):  {rank_info['effective_rank_S']:.2f} / {rank_info['max_possible_rank']} max possible")
    print(f"Effective rank (S^2, EventCLR-style): {rank_info['effective_rank_S2']:.2f} / {rank_info['max_possible_rank']} max possible")
    print(f"Top-1 singular value variance fraction: {rank_info['top1_singular_value_variance_fraction']:.3f}")
    print(f"Per-dim variance: mean={var_info['mean_variance']:.6f}, "
          f"min={var_info['min_variance']:.6f}, max={var_info['max_variance']:.6f}")
    print(f"Fraction of dims with near-zero variance: {var_info['fraction_near_zero_variance']:.2%}")

    collapse_suspected = (
        rank_info["degenerate_all_zero"]
        or rank_info["top1_singular_value_variance_fraction"] > 0.9
        or var_info["fraction_near_zero_variance"] > 0.5
    )
    verdict = "COLLAPSE SUSPECTED" if collapse_suspected else "no obvious collapse signature"
    print(f"Verdict: {verdict}")
    return {"rank": rank_info, "variance": var_info, "collapse_suspected": collapse_suspected}
