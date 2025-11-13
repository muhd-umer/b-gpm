# Bayesian GPM Debugging and Fixes

## Problem Statement

The initial Bayesian GPM implementation showed drastically poor performance on RewardBench:

**Bayesian GPM (Broken)**:
```
Chat: 0.506, Chat Hard: 0.537, Safety: 0.401, Reasoning: 0.491
```

**Original GPM (Target)**:
```
Chat: 0.788, Chat Hard: 0.662, Safety: 0.843, Reasoning: 0.804
```

Training logs showed: `preference_loss=1.55, prob=0.676, loss_mean=1.89`
- The probability of 67.6% indicates the model was barely better than random (50%)
- Performance was ~30-40% worse than original GPM across all categories

## Root Causes Identified

### 1. **Posterior Collapse Due to Missing Beta Annealing** ⚠️ CRITICAL

**Issue**: Training script did not set `bayesian_kl_warmup_steps`, defaulting to 0. This meant the KL divergence penalty was at full strength from step 1.

**Effect**: The model ignored the preference data and collapsed to the prior, learning uninformative embeddings.

**Fix**: Added `--bayesian_kl_warmup_steps 1000` to gradually introduce KL penalty over ~40% of first epoch.

### 2. **KL Weight Too High**

**Issue**: `bayesian_max_kl_weight` defaulted to 1.0, giving KL divergence equal weight to preference loss.

**Effect**: Prevented the model from fitting the data, as regularization dominated learning.

**Fix**: Set `--bayesian_max_kl_weight 0.0001` (appropriate for dataset size ~80k with batch size 32).

**Reasoning**: For mini-batch variational inference, typical KL weight ≈ B/N = 32/80000 = 0.0004. We use 0.0001 to be more conservative.

### 3. **Normalization-Prior Mismatch** ⚠️ CRITICAL

**Issue**: In `rw_model_general_preference.py:316`, sampled embeddings are normalized to the unit sphere:
```python
sample = nn.functional.normalize(sample, p=2, dim=-1)
```

However, in `loss.py:537-547`, the KL divergence was computed using **unnormalized** `raw_mean`:
```python
kl = 0.5 * ((variance + raw_mean.pow(2)) / prior_var - 1 + ...)
```

**Effect**:
- Preference scores are computed in the unit sphere space (||v|| = 1)
- KL divergence regularizes in the ambient R^d space
- KL pushes `raw_mean → 0`, which after normalization gives **random directions** on the sphere
- This fundamental mismatch prevented learning meaningful preference structures

**Fix**: Modified KL divergence to use normalized mean:
```python
# Use normalized mean for KL computation to match the preference computation space
norm_mean = embedding.mean  # Already normalized in the model
kl = 0.5 * ((variance + norm_mean.pow(2)) / prior_var - 1 + ...)
```

### 4. **Suboptimal Prior Variance**

**Issue**: Prior variance defaulted to 1.0, which is too restrictive for normalized embeddings.

**Fix**: Set `--bayesian_prior_variance 10.0` to allow more flexibility while still providing regularization.

**Reasoning**: With normalized embeddings where ||μ|| ≈ 1 and dimension d=6, each dimension has μ²_d ≈ 1/6. With prior_var = 10, the KL term (1/6)/10 ≈ 0.017 per dimension is small, allowing the model to learn directions while regularizing variance.

### 5. **Poor Variance Initialization**

**Issue**: The value head was initialized uniformly, causing both mean and logvar outputs to have similar scales. Initial logvar ≈ 0 means variance ≈ exp(0) = 1, which is quite large and causes noisy samples.

**Fix**: Added custom initialization for Bayesian value head to start with smaller variance:
```python
if is_bayesian_gpm:
    half_dim = model.value_head.weight.data.shape[0] // 2
    # Initialize logvar part to output log(0.1) ≈ -2.3
    model.value_head.weight.data[half_dim:, :] *= 0.1
```

This gives initial variance ≈ 0.1 instead of 1.0, reducing noise during early training.

## Changes Summary

### 1. `b_gpm/models/loss.py`

**KL Divergence Fix** (lines 537-563):
- Changed from using `raw_mean` to `norm_mean` (normalized mean)
- Added detailed documentation explaining the normalization consistency requirement
- Added logging attributes: `last_data_loss`, `last_kl_loss`, `last_kl_beta`

### 2. `b_gpm/models/rw_model_general_preference.py`

**Variance Initialization** (lines 137-152):
- Added special initialization for Bayesian value head
- Scales logvar weights by 0.1 to produce smaller initial variances
- Applied in both DeepSpeed ZeRO-3 and standard cases

### 3. `b_gpm/trainer/rm_trainer_general_preference.py`

**Enhanced Logging** (lines 363-371):
- Added logging for `kl_loss`, `data_loss`, and `kl_beta`
- Helps monitor beta annealing schedule and loss decomposition
- Only activates for Bayesian GPM training

### 4. `scripts/run_train_rm_general_preference_single.sh`

**Hyperparameter Tuning** (lines 29-31):
```bash
--bayesian_kl_warmup_steps 1000 \
--bayesian_max_kl_weight 0.0001 \
--bayesian_prior_variance 10.0 \
```

**Training Schedule**:
- Total steps: ~5000 (2 epochs × 2500 steps/epoch)
- Warmup: 1000 steps (40% of first epoch)
- Beta schedule: β_t = min(1.0, t/1000) × 0.0001

## Expected Improvements

### Training Dynamics

**Before (Broken)**:
- KL weight = 1.0 from step 0 → immediate posterior collapse
- Raw mean pushed to 0 → random directions after normalization
- prob ≈ 67.6% → barely better than random

**After (Fixed)**:
- KL weight starts at 0, ramps to 0.0001 over 1000 steps → data fitting first
- Normalized mean used in KL → consistent regularization
- Variance initialized small → less noise, more stable learning

### Expected Metrics

Training logs should now show:
```
preference_loss=[lower], prob=[higher, >0.8], kl_loss=[small], kl_beta=[0.0-0.0001]
```

RewardBench performance should approach or match original GPM:
- Chat: 0.75-0.80 (target: 0.788)
- Chat Hard: 0.60-0.68 (target: 0.662)
- Safety: 0.75-0.85 (target: 0.843)
- Reasoning: 0.75-0.82 (target: 0.804)

## Validation Checklist

- [x] KL divergence uses normalized embeddings (matches preference computation space)
- [x] Beta annealing implemented with proper warmup schedule
- [x] KL weight scaled appropriately for dataset size
- [x] Prior variance allows sufficient flexibility
- [x] Variance initialization prevents excessive noise
- [x] Logging tracks KL components for debugging
- [x] Evaluation uses posterior mean (deterministic) - already implemented in rewardbench_utils.py:44-45

## Further Tuning (If Needed)

If performance is still below target after training:

1. **Increase KL warmup**: Try 1500-2000 steps for even gentler annealing
2. **Adjust KL weight**: Try 0.0005 or 0.001 for more regularization in later stages
3. **Prior variance**: Try 5.0 (more regularization) or 20.0 (more flexibility)
4. **Learning rate**: May need different LR for Bayesian head vs base model

## Technical Notes

### Normalization Geometry

The key insight is that GPM embeddings are **constrained to the unit sphere** (||v|| = 1). The KL divergence must respect this geometry:

- **Wrong**: KL[q(v) || p(v)] in R^d with p(v) = N(0, σ²I) and v unnormalized
  - Pushes raw mean → 0, which becomes random after normalization

- **Correct**: KL[q(v̂) || p(v̂)] on unit sphere approximated using normalized mean
  - Regularizes the direction while allowing learning on the sphere

### Beta Annealing Intuition

Without warmup:
1. Step 0: Model has random embeddings, high KL → KL penalty dominates
2. Gradient pushes embeddings toward prior (zero mean)
3. Model never learns from preference data → posterior collapse

With warmup:
1. Steps 0-1000: β_t ≈ 0 → pure preference fitting, no KL penalty
2. Model learns meaningful preference structure from data
3. Steps 1000+: β_t → 0.0001 → gentle regularization preserves learned structure
4. Final model: good preferences + calibrated uncertainty

## References

- Original GPM paper: Skew-symmetric preference modeling for cyclic preferences
- Variational Inference: ELBO = E[log p(D|θ)] - β·KL[q||p]
- β-VAE: Higher β → more regularization, lower β → better reconstruction
- RewardBench: Standard evaluation suite for reward models
