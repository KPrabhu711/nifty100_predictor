# Research Upgrades

This note documents the upgrades made after reviewing `../docs/document.pdf`.

## Objective Alignment

The PDF explicitly recommends a final stock-selection score of the form:

`Score_i = w1 * return_head + w2 * rank_head + w3 * direction_signal`

The pipeline now implements this idea with validation-tuned `alpha_score`
selection per fold.

## Changes Made

1. Added validation-tuned `alpha_score` blending of:
   - `pred_return`
   - `pred_rank`
   - `pred_dir_score`
2. Added `score_rank_ic_mean` so model selection can target the portfolio score,
   not only the raw ranking head.
3. Switched early stopping from raw `rank_ic` to `score_rank_ic`.
4. Added recency-weighted training using exponential half-life over training
   dates inside each walk-forward fold.
5. Extended saved predictions with direction probabilities and signed direction
   score for better score construction.
6. Added alpha-weight summary plotting across folds.

## Why These Changes

- The document focuses on portfolio relevance, not just head-wise regression.
- Prior runs showed that the best portfolio score was not always a single head.
- Recent-regime instability suggested a need for recency-aware optimization.

## Recommended Run Mode

- Keep multi-task training.
- Use `alpha_score` for final portfolio construction.
- Judge runs primarily by:
  - `score_rank_ic_mean`
  - `sharpe_ratio`
  - `spread_mean`
  - `max_drawdown`
