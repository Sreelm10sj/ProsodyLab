import numpy as np
from scipy.optimize import linprog
import pywt
import logging

logger = logging.getLogger(__name__)

def solve_mae_lp(t, x, P, constraint_val=None, constraint_t=None, cache=None):
    """
    Solves the MAE LP for a single segment given time indices `t` and data `x`.
    
    Implementation of Yarra & Ghosh 2021, Section 3.3 (Eq 2, 3, 4).
    The cost function is the Sum of Absolute Errors (SAE) to satisfy the 
    additive cost requirement of Dynamic Programming.
    
    Args:
        t (np.ndarray): 1D array of global time indices for this segment.
        x (np.ndarray): 1D array of pitch values for this segment.
        P (int): Polynomial order.
        constraint_val (float, optional): The exact value the polynomial must equal at constraint_t.
        constraint_t (float, optional): The time index where the constraint applies.
        
    Returns:
        tuple: (SAE, coefficients) or (np.inf, None) if infeasible.
    """
    N_seg = len(t)
    if cache is not None:
        key = (t[0], t[-1], P)
        if key not in cache:
            A = np.vander(t, P + 1, increasing=True)
            I = np.eye(N_seg)
            D_mat_new = np.block([
                [A, -I],
                [-A, -I]
            ])
            f_new = np.concatenate([np.zeros(P + 1), np.ones(N_seg)])
            bounds_new = [(None, None)] * (P + 1) + [(0, None)] * N_seg
            cache[key] = (D_mat_new, f_new, bounds_new)
        D_mat, f, bounds = cache[key]
    else:
        # Vandermonde matrix: columns are t^0, t^1, ..., t^P
        A = np.vander(t, P + 1, increasing=True)
        # LP constraints matrix D (Eq 4 in 2021 paper)
        I = np.eye(N_seg)
        D_mat = np.block([
            [A, -I],
            [-A, -I]
        ])
        # Objective vector f
        f = np.concatenate([np.zeros(P + 1), np.ones(N_seg)])
        # Bounds: coefficients are unbounded, theta errors are non-negative
        bounds = [(None, None)] * (P + 1) + [(0, None)] * N_seg
    
    # Target vector y (Eq 4 in 2021 paper)
    y = np.concatenate([x, -x])
    
    # Continuity constraint for DP
    A_eq = None
    b_eq = None
    if constraint_val is not None and constraint_t is not None:
        # poly(constraint_t) = sum(alpha_p * constraint_t^p) = constraint_val
        h = np.zeros(P + 1 + N_seg)
        for p in range(P + 1):
            h[p] = constraint_t ** p
        A_eq = h.reshape(1, -1)
        b_eq = [constraint_val]
        
    # Solve LP using HIGHS
    res = linprog(c=f, A_ub=D_mat, b_ub=y, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        return np.inf, None
        
    phi = res.x
    alpha = phi[:P + 1]
    # The sum of theta variables is exactly the SAE
    sae = np.sum(phi[P + 1:]) 
    
    return sae, alpha

def dp_pitch_stylization(x, K, P, cache=None):
    """
    Finds the optimal K piecewise polynomial segments of order P to approximate x.
    
    Implementation of the Dynamic Programming framework from 
    Ghosh & Narayanan 2009, Section 3.
    
    Args:
        x (np.ndarray): The full pitch contour (0-indexed internally, mapped to 1-indexed mathematically).
        K (int): Number of segments.
        P (int): Polynomial order.
        
    Returns:
        tuple: (eta, polys, total_error)
            eta: Optimal boundary indices [eta_0, ..., eta_K] (1-indexed).
            polys: List of tuples (s, r, coefficients).
            total_error: The minimum total SAE.
    """
    N = len(x)
    # Global time indices: 1 to N
    t_global = np.arange(1, N + 1)
    
    # DP Tables (size (K+1) x (N+1) to allow direct 1-based indexing)
    # D_k(r): minimum SAE for k segments up to r
    D = np.full((K + 1, N + 1), np.inf)
    # beta_k(r): the optimal polynomial value evaluated at r
    beta = np.zeros((K + 1, N + 1))
    # xi_k(r): backtracking pointer storing the best boundary s
    xi = np.zeros((K + 1, N + 1), dtype=int)
    
    # --- 1. Initialization (k = 1) ---
    for r in range(P + 1, N + 1):
        t_seg = t_global[0:r]
        x_seg = x[0:r]
        
        sae, alpha = solve_mae_lp(t_seg, x_seg, P, cache=cache)
        if alpha is not None:
            D[1, r] = sae
            xi[1, r] = 1
            beta[1, r] = sum(alpha[p] * (r ** p) for p in range(P + 1))
            
    # --- 2. Iteration (k = 2 to K) ---
    for k in range(2, K + 1):
        for r in range(k * P + 1, N + 1):
            min_err = np.inf
            best_s = -1
            best_beta_r = 0.0
            
            for s in range((k - 1) * P + 1, r - P + 1):
                if D[k - 1, s] == np.inf:
                    continue
                    
                t_seg = t_global[s - 1:r]
                x_seg = x[s - 1:r]
                
                constraint_val = beta[k - 1, s]
                sae, alpha = solve_mae_lp(t_seg, x_seg, P, constraint_val=constraint_val, constraint_t=s, cache=cache)
                
                if alpha is not None:
                    total_err = D[k - 1, s] + sae
                    if total_err < min_err:
                        min_err = total_err
                        best_s = s
                        best_beta_r = sum(alpha[p] * (r ** p) for p in range(P + 1))
                        
            D[k, r] = min_err
            xi[k, r] = best_s
            beta[k, r] = best_beta_r

    # --- 3. Termination and Backtracking ---
    eta = np.zeros(K + 1, dtype=int)
    eta[K] = N
    
    for k in range(K, 1, -1):
        eta[k - 1] = xi[k, eta[k]]
    eta[0] = 1
    
    # --- 4. Final Polynomial Extraction ---
    polys = []
    
    s, r = eta[0], eta[1]
    t_seg = t_global[s - 1:r]
    x_seg = x[s - 1:r]
    _, alpha = solve_mae_lp(t_seg, x_seg, P, cache=cache)
    if alpha is None: # Should not happen if feasible
        alpha = np.zeros(P+1)
    polys.append((s, r, alpha))
    last_val = sum(alpha[p] * (r ** p) for p in range(P + 1))
    
    for k in range(2, K + 1):
        s = eta[k - 1]
        r = eta[k]
        t_seg = t_global[s - 1:r]
        x_seg = x[s - 1:r]
        _, alpha = solve_mae_lp(t_seg, x_seg, P, constraint_val=last_val, constraint_t=s, cache=cache)
        if alpha is None:
            alpha = np.zeros(P+1)
        polys.append((s, r, alpha))
        last_val = sum(alpha[p] * (r ** p) for p in range(P + 1))
        
    return eta, polys, D[K, N]

def compute_wavelet_k(f0_segment: np.ndarray) -> int:
    """
    Computes K based on the number of extrema in the Level 3 wavelet decomposition.
    """
    try:
        # Use Daubechies 10 ('db10') at level 3
        coeffs = pywt.wavedec(f0_segment, 'db10', level=3)
        # We use the approximation coefficients (cA3) because stylization
        # aims to capture macro-prosody (the major inflection points of the smoothed signal).
        # Detail coefficients would yield too many extrema (noise).
        cA3 = coeffs[0]
        
        # Count extrema (local maxima and minima)
        diff = np.diff(cA3)
        zero_crossings = np.where(np.diff(np.sign(diff)))[0]
        extrema_count = len(zero_crossings)
        
        # Paper: K-1 = number of extrema
        K = extrema_count + 1
        return max(1, K)
    except Exception as e:
        logger.warning(f"Wavelet decomposition failed ({e}). Falling back to K=1.")
        return 1

def stylize_region(segment_f0: np.ndarray, P: int = 1) -> tuple[np.ndarray, int]:
    """
    Applies DP-MAE stylization to a single continuous voiced region.
    """
    K = compute_wavelet_k(segment_f0)
    
    if len(segment_f0) < (P + 1) * K:
        # If segment is too short for K segments of order P, fall back to K=1
        K = 1
        
    if len(segment_f0) < P + 1:
        # Too short for even 1 segment, just return the raw segment
        return segment_f0.copy(), 1
        
    mae_lp_cache = {}
    eta, polys, _ = dp_pitch_stylization(segment_f0, K, P, cache=mae_lp_cache)
    
    stylized = np.zeros(len(segment_f0))
    for s, r, alpha in polys:
        t_seg = np.arange(s, r + 1)
        vals = sum(alpha[p] * (t_seg ** p) for p in range(P + 1))
        # Mathematically s and r are 1-indexed relative to segment_f0
        stylized[s - 1:r] = vals
        
    return stylized, K

def stylize_pitch_contour(pitch_f0: np.ndarray, P: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Identifies voiced regions in the raw pitch contour and applies DP-MAE stylization to each.
    """
    stylized_f0 = np.full_like(pitch_f0, np.nan)
    k_array = np.full_like(pitch_f0, np.nan)
    
    voiced = ~np.isnan(pitch_f0) & (pitch_f0 > 0)
    
    current_start = -1
    for i, is_voiced in enumerate(voiced):
        if is_voiced:
            if current_start == -1:
                current_start = i
        else:
            if current_start != -1:
                segment = pitch_f0[current_start:i]
                stylized_segment, used_k = stylize_region(segment, P)
                stylized_f0[current_start:i] = stylized_segment
                k_array[current_start:i] = used_k
                current_start = -1
                
    if current_start != -1:
        segment = pitch_f0[current_start:]
        stylized_segment, used_k = stylize_region(segment, P)
        stylized_f0[current_start:] = stylized_segment
        k_array[current_start:] = used_k
        
    return stylized_f0, k_array, P
