import numpy as np


def compute_variance_vectorized(mu_i, mu_j, sigma_i, sigma_j):
    r"""Compute variance of the preference score via vectorized operations.

    Given preference score :math:`s = \langle R^\top v_i, v_j \rangle` with
    :math:`v_i \sim \mathcal{N}(\mu_i, \operatorname{diag}(\sigma_i^2))` and
    :math:`v_j \sim \mathcal{N}(\mu_j, \operatorname{diag}(\sigma_j^2))`.

    Args:
        mu_i: Mean vector of embedding :math:`i`, shape ``(2k,)``.
        mu_j: Mean vector of embedding :math:`j`, shape ``(2k,)``.
        sigma_i: Standard deviation vector of embedding :math:`i`, shape ``(2k,)``.
        sigma_j: Standard deviation vector of embedding :math:`j`, shape ``(2k,)``.

    Returns:
        float: Variance of the preference score.

    Notes:
        .. math::
            \operatorname{Var}[s] = \langle \mu_i^{\odot 2}, S \sigma_j^{\odot 2} \rangle
            + \langle \mu_j^{\odot 2}, S \sigma_i^{\odot 2} \rangle
            + \langle \sigma_i^{\odot 2}, S \sigma_j^{\odot 2} \rangle,

        where :math:`S` swaps entries within each consecutive pair.

    Examples:
        >>> mu_i = np.array([1.0, 2.0, 3.0, 4.0])
        >>> mu_j = np.array([0.5, 1.5, 2.5, 3.5])
        >>> sigma_i = np.array([0.1, 0.2, 0.3, 0.4])
        >>> sigma_j = np.array([0.15, 0.25, 0.35, 0.45])
        >>> compute_variance_vectorized(mu_i, mu_j, sigma_i, sigma_j)
    """
    # create swap indices: [1, 0, 3, 2, ...]
    swap_idx = np.arange(len(mu_i))
    swap_idx[::2] += 1
    swap_idx[1::2] -= 1

    # compute squared standard deviations
    sigma_i_sq = sigma_i**2
    sigma_j_sq = sigma_j**2

    # swap pairs in variance vectors
    sigma_j_sq_swapped = sigma_j_sq[swap_idx]
    sigma_i_sq_swapped = sigma_i_sq[swap_idx]

    var = (
        np.sum(mu_i**2 * sigma_j_sq_swapped)
        + np.sum(mu_j**2 * sigma_i_sq_swapped)
        + np.sum(sigma_i_sq * sigma_j_sq_swapped)
    )

    return var


def compute_variance_explicit(mu_i, mu_j, sigma_i, sigma_j, k=None):
    r"""Compute variance using explicit block-wise iteration.

    Args:
        mu_i: Mean vector of embedding :math:`i`, shape ``(2k,)``.
        mu_j: Mean vector of embedding :math:`j`, shape ``(2k,)``.
        sigma_i: Standard deviation vector of embedding :math:`i`, shape ``(2k,)``.
        sigma_j: Standard deviation vector of embedding :math:`j`, shape ``(2k,)``.
        k: Optional number of two-dimensional blocks.

    Returns:
        float: Variance of the preference score.
    """
    if k is None:
        k = len(mu_i) // 2

    var_total = 0.0

    # sum over k blocks
    for b in range(k):
        idx0 = 2 * b
        idx1 = 2 * b + 1

        # note: indices are swapped within each block
        var_total += mu_i[idx0] ** 2 * sigma_j[idx1] ** 2
        var_total += mu_i[idx1] ** 2 * sigma_j[idx0] ** 2
        var_total += mu_j[idx0] ** 2 * sigma_i[idx1] ** 2
        var_total += mu_j[idx1] ** 2 * sigma_i[idx0] ** 2
        var_total += sigma_i[idx0] ** 2 * sigma_j[idx1] ** 2
        var_total += sigma_i[idx1] ** 2 * sigma_j[idx0] ** 2

    return var_total


def compute_variance_matrix_form(mu_i, mu_j, sigma_i, sigma_j):
    r"""Compute variance using the swap-permutation matrix formulation.

    Args:
        mu_i: Mean vector of embedding :math:`i`, shape ``(2k,)``.
        mu_j: Mean vector of embedding :math:`j`, shape ``(2k,)``.
        sigma_i: Standard deviation vector of embedding :math:`i`, shape ``(2k,)``.
        sigma_j: Standard deviation vector of embedding :math:`j`, shape ``(2k,)``.

    Returns:
        float: Variance of the preference score.

    Notes:
        .. math::
            \operatorname{Var}[s] = (\mu_i^{\odot 2})^\top (S \sigma_j^{\odot 2})
            + (\mu_j^{\odot 2})^\top (S \sigma_i^{\odot 2})
            + (\sigma_i^{\odot 2})^\top (S \sigma_j^{\odot 2}).
    """
    dim = len(mu_i)
    k = dim // 2

    # build swap permutation matrix s
    S = np.zeros((dim, dim))
    for b in range(k):
        idx0 = 2 * b
        idx1 = 2 * b + 1
        S[idx0, idx1] = 1
        S[idx1, idx0] = 1

    # compute squared terms
    sigma_i_sq = sigma_i**2
    sigma_j_sq = sigma_j**2
    mu_i_sq = mu_i**2
    mu_j_sq = mu_j**2

    # apply formula
    var = (
        np.dot(mu_i_sq, S @ sigma_j_sq)
        + np.dot(mu_j_sq, S @ sigma_i_sq)
        + np.dot(sigma_i_sq, S @ sigma_j_sq)
    )

    return var


def verify_implementation(dim=8, n_tests=10, mc_samples=100000, verbose=True):
    """Verify the variance implementations against Monte Carlo simulation.

    Args:
        dim: Even dimensionality of the embeddings.
        n_tests: Number of random test cases.
        mc_samples: Number of Monte Carlo samples per test.
        verbose: Whether to emit per-test diagnostics.

    Returns:
        dict: Aggregated verification metrics.
    """
    assert dim % 2 == 0, "Dimension must be even"
    k = dim // 2

    # build r^⊤ matrix
    R = np.zeros((dim, dim))
    for b in range(k):
        R[2 * b, 2 * b + 1] = 1
        R[2 * b + 1, 2 * b] = -1

    errors_vec = []
    errors_exp = []
    errors_mat = []

    for test_num in range(n_tests):
        # random test parameters
        mu_i = np.random.randn(dim)
        mu_j = np.random.randn(dim)
        sigma_i = np.abs(np.random.randn(dim)) + 0.5
        sigma_j = np.abs(np.random.randn(dim)) + 0.5

        # compute using different methods
        var_vec = compute_variance_vectorized(mu_i, mu_j, sigma_i, sigma_j)
        var_exp = compute_variance_explicit(mu_i, mu_j, sigma_i, sigma_j, k)
        var_mat = compute_variance_matrix_form(mu_i, mu_j, sigma_i, sigma_j)

        # monte carlo ground truth
        v_i_samples = np.random.randn(mc_samples, dim) * sigma_i + mu_i
        v_j_samples = np.random.randn(mc_samples, dim) * sigma_j + mu_j
        s_samples = np.sum(v_i_samples @ R * v_j_samples, axis=1)
        var_mc = np.var(s_samples, ddof=1)

        # compute relative errors
        rel_err_vec = abs(var_vec - var_mc) / abs(var_mc)
        rel_err_exp = abs(var_exp - var_mc) / abs(var_mc)
        rel_err_mat = abs(var_mat - var_mc) / abs(var_mc)

        errors_vec.append(rel_err_vec)
        errors_exp.append(rel_err_exp)
        errors_mat.append(rel_err_mat)

        if verbose:
            status = "[p]" if rel_err_vec < 0.025 else "[f]"
            print(
                f"Test {test_num+1:2d}: Closed={var_vec:6.4f}, MC={var_mc:6.4f}, "
                f"Error={rel_err_vec:.4f} {status}"
            )

    results = {
        "mean_error_vectorized": np.mean(errors_vec),
        "max_error_vectorized": np.max(errors_vec),
        "mean_error_explicit": np.mean(errors_exp),
        "max_error_explicit": np.max(errors_exp),
        "mean_error_matrix": np.mean(errors_mat),
        "max_error_matrix": np.max(errors_mat),
        "all_pass": np.max(errors_vec) < 0.025,
    }

    if verbose:
        summary = (
            f"\nmean relative error: {results['mean_error_vectorized']:.4%}\n"
            f"max relative error: {results['max_error_vectorized']:.4%}\n"
            f"all tests pass (<2.5% error): {results['all_pass']}\n"
        )
        print(summary)

    return results


if __name__ == "__main__":
    results = verify_implementation(dim=8, n_tests=10, mc_samples=500000)
