import time
import numpy as np
import pywt
import whistress_service
from pitch_analysis import extract_pitch
from scipy.optimize import linprog

# Counters
stats = {
    'total_voiced_regions': 0,
    'region_stats': [],
}

def solve_mae_lp_instrumented(t, x, P, constraint_val=None, constraint_t=None, cache=None):
    stats['lp_solves'] += 1
    
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
        A = np.vander(t, P + 1, increasing=True)
        I = np.eye(N_seg)
        D_mat = np.block([
            [A, -I],
            [-A, -I]
        ])
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

def dp_pitch_stylization_instrumented(x, K, P, cache=None):
    N = len(x)
    t_global = np.arange(1, N + 1)
    
    D = np.full((K + 1, N + 1), np.inf)
    beta = np.zeros((K + 1, N + 1))
    xi = np.zeros((K + 1, N + 1), dtype=int)
    
    # 1. Initialization
    for r in range(P + 1, N + 1):
        t_seg = t_global[0:r]
        x_seg = x[0:r]
        
        sae, alpha = solve_mae_lp_instrumented(t_seg, x_seg, P, cache=cache)
        if alpha is not None:
            D[1, r] = sae
            xi[1, r] = 1
            beta[1, r] = sum(alpha[p] * (r ** p) for p in range(P + 1))
            
    # 2. Iteration
    for k in range(2, K + 1):
        for r in range(k * P + 1, N + 1):
            min_err = np.inf
            best_s = -1
            best_beta_r = 0.0
            
            for s in range((k - 1) * P + 1, r - P + 1):
                stats['dp_states_evaluated'] += 1
                if D[k - 1, s] == np.inf:
                    continue
                    
                t_seg = t_global[s - 1:r]
                x_seg = x[s - 1:r]
                
                constraint_val = beta[k - 1, s]
                sae, alpha = solve_mae_lp_instrumented(t_seg, x_seg, P, constraint_val=constraint_val, constraint_t=s, cache=cache)
                
                if alpha is not None:
                    total_err = D[k - 1, s] + sae
                    if total_err < min_err:
                        min_err = total_err
                        best_s = s
                        best_beta_r = sum(alpha[p] * (r ** p) for p in range(P + 1))
                        
            D[k, r] = min_err
            xi[k, r] = best_s
            beta[k, r] = best_beta_r

    # 3. Termination and Backtracking
    eta = np.zeros(K + 1, dtype=int)
    eta[K] = N
    for k in range(K, 1, -1):
        eta[k - 1] = xi[k, eta[k]]
    eta[0] = 1
    
    polys = []
    s, r = eta[0], eta[1]
    t_seg = t_global[s - 1:r]
    x_seg = x[s - 1:r]
    _, alpha = solve_mae_lp_instrumented(t_seg, x_seg, P, cache=cache)
    if alpha is None: alpha = np.zeros(P+1)
    polys.append((s, r, alpha))
    last_val = sum(alpha[p] * (r ** p) for p in range(P + 1))
    
    for k in range(2, K + 1):
        s = eta[k - 1]
        r = eta[k]
        t_seg = t_global[s - 1:r]
        x_seg = x[s - 1:r]
        _, alpha = solve_mae_lp_instrumented(t_seg, x_seg, P, constraint_val=last_val, constraint_t=s, cache=cache)
        if alpha is None: alpha = np.zeros(P+1)
        polys.append((s, r, alpha))
        last_val = sum(alpha[p] * (r ** p) for p in range(P + 1))
        
    return eta, polys, D[K, N]

def compute_wavelet_k(f0_segment: np.ndarray) -> int:
    try:
        coeffs = pywt.wavedec(f0_segment, 'db10', level=3)
        cA3 = coeffs[0]
        diff = np.diff(cA3)
        zero_crossings = np.where(np.diff(np.sign(diff)))[0]
        K = len(zero_crossings) + 1
        return max(1, K)
    except:
        return 1

def stylize_region_instrumented(segment_f0: np.ndarray, P: int = 1) -> np.ndarray:
    t_start = time.perf_counter()
    K = compute_wavelet_k(segment_f0)
    
    if len(segment_f0) < (P + 1) * K: K = 1
    if len(segment_f0) < P + 1: return segment_f0.copy()
        
    mae_lp_cache = {}
    stats['lp_solves'] = 0
    stats['dp_states_evaluated'] = 0
    
    eta, polys, _ = dp_pitch_stylization_instrumented(segment_f0, K, P, cache=mae_lp_cache)
    
    t_end = time.perf_counter()
    region_time = t_end - t_start
    
    stats['region_stats'].append({
        'frames': len(segment_f0),
        'K': K,
        'dp_states': stats['dp_states_evaluated'],
        'lp_solves': stats['lp_solves'],
        'time': region_time
    })
    
    stylized = np.zeros(len(segment_f0))
    for s, r, alpha in polys:
        t_seg = np.arange(s, r + 1)
        vals = sum(alpha[p] * (t_seg ** p) for p in range(P + 1))
        stylized[s - 1:r] = vals
    return stylized

def profile_audio(filename):
    print(f"\n--- Profiling {filename} ---")
    stats['total_voiced_regions'] = 0
    stats['region_stats'] = []
    
    with open(filename, "rb") as f:
        audio_bytes = f.read()
        
    service = whistress_service.WhiStressService()
    audio_dict = service.process_audio(audio_bytes)
    pitch_times, pitch_f0 = extract_pitch(audio_dict["array"], sr=audio_dict["sampling_rate"])
    
    total_audio_duration = len(audio_dict["array"]) / audio_dict["sampling_rate"]
    
    t0 = time.perf_counter()
    
    stylized_f0 = np.full_like(pitch_f0, np.nan)
    voiced = ~np.isnan(pitch_f0) & (pitch_f0 > 0)
    
    current_start = -1
    for i, is_voiced in enumerate(voiced):
        if is_voiced:
            if current_start == -1: current_start = i
        else:
            if current_start != -1:
                stats['total_voiced_regions'] += 1
                segment = pitch_f0[current_start:i]
                stylized_f0[current_start:i] = stylize_region_instrumented(segment, 1)
                current_start = -1
                
    if current_start != -1:
        stats['total_voiced_regions'] += 1
        segment = pitch_f0[current_start:]
        stylized_f0[current_start:] = stylize_region_instrumented(segment, 1)
        
    total_time = time.perf_counter() - t0
    
    # Calculate metrics
    total_voiced_frames = sum(r['frames'] for r in stats['region_stats'])
    total_lp_solves = sum(r['lp_solves'] for r in stats['region_stats'])
    total_dp_states = sum(r['dp_states'] for r in stats['region_stats'])
    
    frame_step = 0.01  # Assuming 10ms frame step for Praat default
    total_voiced_duration = total_voiced_frames * frame_step
    
    print(f"Total audio duration: {total_audio_duration:.2f} s")
    print(f"Total voiced duration: {total_voiced_duration:.2f} s")
    print(f"Number of voiced regions: {stats['total_voiced_regions']}")
    
    if stats['region_stats']:
        max_len = max(r['frames'] for r in stats['region_stats'])
        avg_len = sum(r['frames'] for r in stats['region_stats']) / len(stats['region_stats'])
        print(f"Maximum voiced-region length: {max_len} frames")
        print(f"Average voiced-region length: {avg_len:.1f} frames")
    
    print(f"Total DP states: {total_dp_states}")
    print(f"Total LP solves: {total_lp_solves}")
    print(f"Total runtime: {total_time:.2f} s")
    
    print("\nPer-Region Statistics:")
    print("Region | Frames | K | DP States | LP Solves | Time (s)")
    print("-" * 60)
    for i, r in enumerate(stats['region_stats']):
        print(f"{i+1:6d} | {r['frames']:6d} | {r['K']:1d} | {r['dp_states']:9d} | {r['lp_solves']:9d} | {r['time']:.3f}")


if __name__ == "__main__":
    profile_audio("example1_football.wav")
    profile_audio("example3.wav")
