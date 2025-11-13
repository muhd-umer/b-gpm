# Bayesian GPM Fixes

## Problem
Initial Bayesian GPM showed dramatically poor performance (Chat: 0.506 vs target: 0.788).

## Root Cause
**Critical**: KL divergence was computed incorrectly. The implementation used `norm_mean` (normalized embeddings on unit sphere) in the KL formula, but the variational distribution is in ambient space before normalization.

## Solution

### 1. Corrected KL Divergence
**Key insight**: The reparameterization is `v = normalize(raw_mu + eps * std)`. The prior and KL should be in ambient space, not on the sphere.

**New implementation**:
- Always regularize variance: `KL = 0.5 * (variance/prior_var - 1 + log(prior_var) - logvar)`
- Optionally regularize mean: Add `0.5 * (raw_mean^2 / prior_var)` if `regularize_mean=True`
- Default: Only variance regularization allows raw_mean to grow freely, giving strong directional preferences after normalization

### 2. Updated Hyperparameters
```bash
--bayesian_prior_variance 0.1      # Small prior forces tight variance
--bayesian_max_kl_weight 0.05      # Moderate variance regularization
--bayesian_kl_warmup_steps 500     # 20% of first epoch
```

**Rationale**:
- Small prior_variance (0.1) + moderate kl_weight (0.05) → controls uncertainty
- No mean regularization → allows learning strong preferences
- Beta annealing prevents posterior collapse

### 3. Optional Mean Regularization
```bash
--bayesian_regularize_mean  # Add flag to also penalize mean magnitude
```
Default is `False` - recommended for most cases.

### 4. Enhanced Logging
Training now logs:
- `bayes_kl`: KL divergence value
- `kl_beta`: Current annealing coefficient

## Files Changed
- `b_gpm/models/loss.py`: Corrected KL formula, separated variance/mean regularization
- `b_gpm/trainer/rm_trainer_general_preference.py`: Added parameter passing and logging
- `b_gpm/models/rw_model_general_preference.py`: Improved variance initialization
- `scripts/train_rm_general_preference.py`: Added `--bayesian_regularize_mean` flag
- `scripts/run_train_rm_general_preference_single.sh`: Updated hyperparameters

## Expected Performance
Should match or approach original GPM: Chat ~0.79, Chat Hard ~0.66, Safety ~0.84, Reasoning ~0.80
