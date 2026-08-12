import numpy as np
from scipy.optimize import linprog
import pywt
import logging

logger = logging.getLogger(__name__)

# --- ORIGINAL MAE LP SOLVER (Preserved for final segment fitting) ---

def solve_mae_lp(t, x, P, constraint_val=None, constraint_t=None, cache=None):
    N_seg = len(t)
    if cache is not None:
        key = (t[0], t[-1], P)
        if key not in cache:
            A = np.vander(t, P + 1, increasing=True)
            I = np.eye(N_seg)
            D_mat_new = np.block([[A, -I], [-A, -I]])
            f_new = np.concatenate([np.zeros(P + 1), np.ones(N_seg)])
            bounds_new = [(None, None)] * (P + 1) + [(0, None)] * N_seg
            cache[key] = (D_mat_new, f_new, bounds_new, A)
        D_mat, f, bounds, A = cache[key]
    else:
        A = np.vander(t, P + 1, increasing=True)
        I = np.eye(N_seg)
        D_mat = np.block([[A, -I], [-A, -I]])
        f = np.concatenate([np.zeros(P + 1), np.ones(N_seg)])
        bounds = [(None, None)] * (P + 1) + [(0, None)] * N_seg
    
    y = np.concatenate([x, -x])
    
    A_eq = None
    b_eq = None
    if constraint_val is not None and constraint_t is not None:
        h = np.zeros(P + 1 + N_seg)
        for p in range(P + 1):
            h[p] = constraint_t ** p
        A_eq = h.reshape(1, -1)
        b_eq = [constraint_val]
        
    res = linprog(c=f, A_ub=D_mat, b_ub=y, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        return np.inf, None
        
    phi = res.x
    alpha = phi[:P + 1]
    sae = np.sum(phi[P + 1:]) 
    return sae, alpha

# --- FAST MSE SOLVER (For DP Boundary Search) ---

def solve_mse_kkt(t, x, P, constraint_val=None, constraint_t=None, cache=None):
    """
    Solves L2 norm polynomial fitting extremely quickly using linear algebra.
    Used during DP search to avoid thousands of LP solves.
    """
    if cache is not None:
        key = (t[0], t[-1], P)
        if key not in cache:
            A = np.vander(t, P + 1, increasing=True)
            cache[key] = A
        A = cache[key]
    else:
        A = np.vander(t, P + 1, increasing=True)
        
    AtA = A.T @ A
    Atx = A.T @ x
    
    if constraint_val is not None and constraint_t is not None:
        C = np.array([[constraint_t**p for p in range(P + 1)]])
        d = np.array([constraint_val])
        
        K_top = np.hstack([2 * AtA, C.T])
        K_bot = np.hstack([C, np.zeros((1, 1))])
        K = np.vstack([K_top, K_bot])
        rhs = np.concatenate([2 * Atx, d])
        
        try:
            sol = np.linalg.solve(K, rhs)
            alpha = sol[:P + 1]
        except np.linalg.LinAlgError:
            return np.inf, None
    else:
        try:
            alpha = np.linalg.solve(AtA, Atx)
        except np.linalg.LinAlgError:
            return np.inf, None
            
    # Compute error. We use Sum of Absolute Errors (SAE) of the MSE-fitted line 
    # to maintain strict objective scale parity, though MSE itself also works.
    # Actually, using MSE as the DP objective is mathematically correct for an L2 fit.
    err = np.sum((A @ alpha - x)**2)
    return err, alpha


def dp_pitch_stylization_fast(x, K, P, cache=None):
    N = len(x)
    t_global = np.arange(1, N + 1)
    
    D = np.full((K + 1, N + 1), np.inf)
    beta = np.zeros((K + 1, N + 1))
    xi = np.zeros((K + 1, N + 1), dtype=int)
    
    mse_cache = {}
    
    # --- 1. Initialization (using MSE) ---
    for r in range(P + 1, N + 1):
        t_seg = t_global[0:r]
        x_seg = x[0:r]
        
        err, alpha = solve_mse_kkt(t_seg, x_seg, P, cache=mse_cache)
        if alpha is not None:
            D[1, r] = err
            xi[1, r] = 1
            beta[1, r] = sum(alpha[p] * (r ** p) for p in range(P + 1))
            
    # --- 2. Iteration (using MSE) ---
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
                err, alpha = solve_mse_kkt(t_seg, x_seg, P, constraint_val=constraint_val, constraint_t=s, cache=mse_cache)
                
                if alpha is not None:
                    total_err = D[k - 1, s] + err
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
    
    # --- 4. Final Polynomial Extraction (Using MAE for robustness) ---
    # We found the boundaries `eta` incredibly fast using MSE. 
    # Now we solve the slow MAE LP exactly once per segment.
    polys = []
    mae_cache = {}
    
    s, r = eta[0], eta[1]
    t_seg = t_global[s - 1:r]
    x_seg = x[s - 1:r]
    _, alpha = solve_mae_lp(t_seg, x_seg, P, cache=mae_cache)
    if alpha is None:
        alpha = np.zeros(P+1)
    polys.append((s, r, alpha))
    last_val = sum(alpha[p] * (r ** p) for p in range(P + 1))
    
    for k in range(2, K + 1):
        s = eta[k - 1]
        r = eta[k]
        t_seg = t_global[s - 1:r]
        x_seg = x[s - 1:r]
        _, alpha = solve_mae_lp(t_seg, x_seg, P, constraint_val=last_val, constraint_t=s, cache=mae_cache)
        if alpha is None:
            alpha = np.zeros(P+1)
        polys.append((s, r, alpha))
        last_val = sum(alpha[p] * (r ** p) for p in range(P + 1))
        
    return eta, polys, D[K, N]


def compute_wavelet_k(f0_segment: np.ndarray) -> int:
    try:
        coeffs = pywt.wavedec(f0_segment, 'db10', level=3)
        cA3 = coeffs[0]
        zero_crossings = np.where(np.diff(np.sign(np.diff(cA3))))[0]
        K = len(zero_crossings) + 1
        return max(1, K)
    except Exception as e:
        logger.warning(f"Wavelet decomposition failed ({e}). Falling back to K=1.")
        return 1


def stylize_region(segment_f0: np.ndarray, P: int = 1) -> tuple[np.ndarray, int]:
    K = compute_wavelet_k(segment_f0)
    
    if len(segment_f0) < (P + 1) * K:
        K = 1
        
    if len(segment_f0) < P + 1:
        return segment_f0.copy(), 1
        
    eta, polys, _ = dp_pitch_stylization_fast(segment_f0, K, P)
    
    stylized = np.zeros(len(segment_f0))
    for s, r, alpha in polys:
        t_seg = np.arange(s, r + 1)
        vals = sum(alpha[p] * (t_seg ** p) for p in range(P + 1))
        stylized[s - 1:r] = vals
        
    return stylized, K


def stylize_pitch_contour(pitch_f0: np.ndarray, P: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
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
