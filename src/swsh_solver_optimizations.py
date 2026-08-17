# -------------------------------------------------------------------
#  (alpha-transformed) Prufer-equation Brent-method eigenvalue finder
# -------------------------------------------------------------------
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import time
import matplotlib.pyplot as plt 
from math import tanh, cos, sin, sqrt


def spheroidal_eigenvalue_mid(s, m, l, c, eps=15, tol=1e-15, plot_history=False):   
    # eps > 10 is good for high vals. Can make it adaptive if you want

    # -----------------------------------------------------------------
    # 1.  Initial parts
    # -----------------------------------------------------------------
    time_start = time.time()
    N  = int(round(l - max(abs(m), abs(s)))) 
    max_step_calibrated = 0.1 / N
    max_iter_calibrated = 10 * N

    def cot_endpoint_def():
        left_point_start = np.arctan2(N, np.abs(m-s))
        right_point_start = - np.arctan2(N, np.abs(m+s)) + (N + 1) * np.pi
        return left_point_start, right_point_start

    # -----------------------------------------------------------------
    # 2.  Prüfer ODE (λ is a parameter)
    # -----------------------------------------------------------------
    def alpha_prime(w, alpha, lam):
        tanhw = tanh(w)
        tanhw_sq = tanhw**2
        sechw_sq = 1.0 - tanhw_sq
        #sechw = sqrt(sechw_sq)
        cosalpha_sq = cos(alpha)**2 # Deprecation warning: no compression from array to scalar in the future... hmmm
        sinalpha_sq = sin(alpha)**2 # Deprecation warning: no compression from array to scalar in the future... hmmm
        f = (lam + ((c**2) * (tanhw_sq)) - (2.0 * c * s * tanhw) + s) * (sechw_sq) - (m + s * tanhw)**2
        return N * (cosalpha_sq) + (f/N) * (sinalpha_sq)
    # make a grid of these values that the integratory can just call?


    # -----------------------------------------------------------------
    # 3.  Residual for the meet‑in‑the‑middle condition
    # -----------------------------------------------------------------
    if plot_history == True:
        solution_history = [] 

    th_left_start, th_right_start = cot_endpoint_def()

    def mid_residual(lam):

        # ---- left‑hand integration (−ε → 0) ----
        Npts = 100 * int(round(np.sqrt(N))) # adaptive? i think this is good?
        xs_left = np.linspace(-eps, 0.0, Npts)   # increasing
        sol_f = solve_ivp(alpha_prime, 
                            t_span=(-eps, 0.0),
                            y0=[th_left_start],
                            args=(lam,),
                            method='RK45',
                            max_step=max_step_calibrated,
                            t_eval=xs_left)
        alpha_left = sol_f.y[0]                       # array of length Npts

        # ---- right‑hand integration (+ε → 0) ----
        xs_right = np.linspace(eps, 0.0, Npts)   # decreasing
        sol_b = solve_ivp(alpha_prime,
                            t_span=(eps, 0.0),
                            y0=[th_right_start],
                            args=(lam,),
                            method='RK45',
                            max_step=max_step_calibrated,
                            t_eval=xs_right)
        alpha_right = sol_b.y[0]

        # keep the data for later plotting
        if plot_history == True:
            solution_history.append((lam, xs_left, alpha_left, xs_right, alpha_right))

        # residual is the jump at the midpoint
        alpha_left_mid  = alpha_left[-1] # value at x = 0-
        alpha_right_mid = alpha_right[-1] # value at x = 0+
        return alpha_left_mid - alpha_right_mid

    # -----------------------------------------------------------------
    # 4.  Bracket the root (exactly as you did before)
    # -----------------------------------------------------------------
    lam0   = l * (l + 1) - s * (s + 1) # spherical guess (switch with WKB?)
    delta  = max(5.0, 5.0 * abs(c)**2) + 0.25 * lam0 # initial half‑width
    lam_lo = lam0 - delta
    lam_hi = lam0 + delta

    # -----------------------------------------------------------------
    # 5.  Brent root‑find
    # -----------------------------------------------------------------
    # https://docs.python.org/3/library/concurrent.futures.html (new guesses for each part?)
    lam = brentq(mid_residual, lam_lo, lam_hi, xtol=tol, rtol=tol, maxiter=max_iter_calibrated)

    # -----------------------------------------------------------------
    # 6.  Plot the stored Prüfer curves (optional)
    # -----------------------------------------------------------------
    time_end = time.time()
    print(f"λ(s={s},m={m},l={l},c={c}) = {lam:.12f}), time={time_end-time_start}")
    if plot_history:
            plt.figure(figsize=(9, 6))
            for lam_guess, xs_l, th_l, xs_r, th_r in solution_history:
                # Merge the two halves so that we obtain a curve on [-1, +1]
                xs_total = np.concatenate([xs_l, xs_r[::-1]])          # left: -1 → 0, right: 0 → +1
                th_total = np.concatenate([th_l, th_r[::-1]])
                plt.plot(xs_total, th_total,
                         label=f"λ≈{lam_guess:.5g}",
                         linewidth=1.2,
                         alpha=0.7)
            plt.xlabel(r"$w$")
            plt.ylabel(r"α(w)")
            plt.title(f"Prüfer‑phase α(w) for every λ‑guess (s={s}, l={l}, m={m}, c={c}, ε={eps}, tol={tol})")
            plt.legend(title=r"λ‑guess", fontsize="small", loc="best")
            plt.grid(True, which="both", ls=":")
            plt.tight_layout()
            plt.show()
            return lam, time_end - time_start #, solution_history
    else:
        return lam, time_end - time_start


# -----------------------------------------------------------------
# Example driver (feel free to change the parameters)
# -----------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: spherical case, s=2, m=3, l=5, c=0  →  λ = l(l+1) - s(s+1) = 30 - 6 = 24
    lam, time_val = spheroidal_eigenvalue_mid(s=2.0, m=3.0, l=1000.0, c=0.0, eps=15, plot_history=False)
    #print("time: ", time_val)
    #print(f"λ(s={s:.1f},m=2,l=3,c=0) = {lam:.12f})")#, (time = {time_end - time_start:.4f} s)")
    #print(time_end - time_start)

    # Test 2: spherical case, s=2, m=5, l=10, c=0  →  λ = 30(30+1) - 6 = 930 - 6 = 924
    #lam2, _ = spheroidal_eigenvalue_mid(s=2.0, m=5.0, l=30.0, c=0.0)
    #print(f"λ(s=2,m=5,l=10,c=0) = {lam2:.12f}")



"""
Examples:
-> λ(s=2.0,m=3.0,l=10.0,c=0.0) = 103.999999998183), time=0.4130859375
-> λ(s=2.0,m=3.0,l=100.0,c=0.0) = 10093.999999895541), time=6.681729078292847
-> λ(s=2.0,m=3.0,l=50.0,c=0.0) = 2543.999999963187), time=3.2522618770599365
-> λ(s=2.0,m=3.0,l=200.0,c=0.0) = 40193.999999679763), time=18.105145692825317
-> λ(s=2.0,m=3.0,l=500.0,c=0.0) = 250493.999998450221), time=45.536524057388306
"""