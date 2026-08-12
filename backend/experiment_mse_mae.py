import time
import numpy as np
import pywt
import whistress_service
from pitch_analysis import extract_pitch
from scipy.optimize import linprog

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
    
    A_eq, b_eq = None, None
    if constraint_val is not None and constraint_t is not None:
        h = np.zeros(P + 1 + N_seg)
        for p in range(P + 1):
            h[p] = constraint_t ** p
        A_eq = h.reshape(1, -1)
        b_eq = [constraint_val]
        
    res = linprog(c=f, A_ub=D_mat, b_ub=y, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        return np.inf, None
    return np.sum(res.x[P + 1:]), res.x[:P + 1]

def solve_mse_kkt(t, x, P, constraint_val=None, constraint_t=None, cache=None):
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
        
        # KKT system for min ||Ax - y||^2 s.t. Cx = d
        # [ 2AtA   C^T ] [ alpha ] = [ 2Atx ]
        # [ C       0  ] [ lambda]   [ d    ]
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
            
    err = np.sum((A @ alpha - x)**2)
    return err, alpha

def compute_wavelet_k(f0_segment: np.ndarray) -> int:
    try:
        coeffs = pywt.wavedec(f0_segment, 'db10', level=3)
        zero_crossings = np.where(np.diff(np.sign(np.diff(coeffs[0]))))[0]
        return max(1, len(zero_crossings) + 1)
    except:
        return 1

# Statistics tracking
agreement_stats = {
    'total_evaluations': 0,
    'exact_matches': 0,
    'displacements': []
}

def dp_pitch_stylization_experiment(x_arr, K, P):
    N = len(x_arr)
    t_global = np.arange(1, N + 1)
    
    # DP Tables for MAE
    D_mae = np.full((K + 1, N + 1), np.inf)
    beta_mae = np.zeros((K + 1, N + 1))
    
    # DP Tables for MSE
    D_mse = np.full((K + 1, N + 1), np.inf)
    beta_mse = np.zeros((K + 1, N + 1))
    
    mae_cache = {}
    mse_cache = {}
    
    # Initialization
    for r in range(P + 1, N + 1):
        t_seg = t_global[0:r]
        x_seg = x_arr[0:r]
        
        sae, alpha_mae = solve_mae_lp(t_seg, x_seg, P, cache=mae_cache)
        if alpha_mae is not None:
            D_mae[1, r] = sae
            beta_mae[1, r] = sum(alpha_mae[p] * (r ** p) for p in range(P + 1))
            
        mse, alpha_mse = solve_mse_kkt(t_seg, x_seg, P, cache=mse_cache)
        if alpha_mse is not None:
            D_mse[1, r] = mse
            beta_mse[1, r] = sum(alpha_mse[p] * (r ** p) for p in range(P + 1))
            
    # Iteration
    for k in range(2, K + 1):
        for r in range(k * P + 1, N + 1):
            
            # Find best_s using MAE
            min_err_mae = np.inf
            best_s_mae = -1
            best_beta_r_mae = 0.0
            
            # Find best_s using MSE
            min_err_mse = np.inf
            best_s_mse = -1
            best_beta_r_mse = 0.0
            
            for s in range((k - 1) * P + 1, r - P + 1):
                t_seg = t_global[s - 1:r]
                x_seg = x_arr[s - 1:r]
                
                # Evaluate MAE
                if D_mae[k - 1, s] != np.inf:
                    sae, a_mae = solve_mae_lp(t_seg, x_seg, P, constraint_val=beta_mae[k - 1, s], constraint_t=s, cache=mae_cache)
                    if a_mae is not None:
                        tot_mae = D_mae[k - 1, s] + sae
                        if tot_mae < min_err_mae:
                            min_err_mae = tot_mae
                            best_s_mae = s
                            best_beta_r_mae = sum(a_mae[p] * (r ** p) for p in range(P + 1))
                            
                # Evaluate MSE
                if D_mse[k - 1, s] != np.inf:
                    mse, a_mse = solve_mse_kkt(t_seg, x_seg, P, constraint_val=beta_mse[k - 1, s], constraint_t=s, cache=mse_cache)
                    if a_mse is not None:
                        tot_mse = D_mse[k - 1, s] + mse
                        if tot_mse < min_err_mse:
                            min_err_mse = tot_mse
                            best_s_mse = s
                            best_beta_r_mse = sum(a_mse[p] * (r ** p) for p in range(P + 1))
                            
            if best_s_mae != -1 and best_s_mse != -1:
                agreement_stats['total_evaluations'] += 1
                if best_s_mae == best_s_mse:
                    agreement_stats['exact_matches'] += 1
                else:
                    agreement_stats['displacements'].append(abs(best_s_mae - best_s_mse))
                    
            D_mae[k, r] = min_err_mae
            beta_mae[k, r] = best_beta_r_mae
            
            D_mse[k, r] = min_err_mse
            beta_mse[k, r] = best_beta_r_mse

def run_experiment(filename):
    print(f"Running MSE/MAE Boundary Experiment on {filename}...")
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    service = whistress_service.WhiStressService()
    audio_dict = service.process_audio(audio_bytes)
    _, pitch_f0 = extract_pitch(audio_dict["array"], sr=audio_dict["sampling_rate"])
    
    voiced = ~np.isnan(pitch_f0) & (pitch_f0 > 0)
    
    current_start = -1
    for i, is_voiced in enumerate(voiced):
        if is_voiced:
            if current_start == -1: current_start = i
        else:
            if current_start != -1:
                segment = pitch_f0[current_start:i]
                K = compute_wavelet_k(segment)
                if len(segment) >= 2 * K and K > 1:
                    # Skip extremely long regions for this experiment to save time,
                    # we just need a statistical sample of boundaries.
                    if len(segment) <= 40:
                        dp_pitch_stylization_experiment(segment, K, 1)
                current_start = -1

if __name__ == "__main__":
    t0 = time.perf_counter()
    run_experiment("example3.wav")
    
    tot = agreement_stats['total_evaluations']
    matches = agreement_stats['exact_matches']
    displacements = agreement_stats['displacements']
    
    print("\n--- EXPERIMENT RESULTS ---")
    print(f"Total (k, r) DP states evaluated: {tot}")
    if tot > 0:
        print(f"Exact Matches (MAE optimal boundary == MSE optimal boundary): {matches} ({matches/tot*100:.1f}%)")
        if displacements:
            print(f"Average displacement when differing: {np.mean(displacements):.2f} frames")
            print(f"Max displacement: {np.max(displacements)} frames")
            
            print("\nDisplacement Distribution:")
            for d in range(1, 6):
                count = sum(1 for x in displacements if x == d)
                print(f"  Off by {d} frames: {count} ({count/tot*100:.1f}%)")
            count_large = sum(1 for x in displacements if x > 5)
            print(f"  Off by >5 frames: {count_large} ({count_large/tot*100:.1f}%)")
    print(f"\nExperiment Runtime: {time.perf_counter() - t0:.2f}s")
