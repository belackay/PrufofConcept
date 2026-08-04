# --------------------------------------------------------------
#  spheroidal eigenvalue via Prüfer (phase) shooting – meet‑in‑the‑middle
#  THIS time... I'm gonna transform it to tanhw. YEAH!!
# --------------------------------------------------------------
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import time


def spheroidal_eigenvalue_mid(s, m, l, c,
                             eps=10,
                             max_iter=100,
                             tol=1e-15,
                             max_step=0.01):
    """
    Compute λ_{slm}(c) with a two‑point shooting method.
    Returns (λ, cpu_time) where the cpu_time is the time spent inside Brent.
    """

    # -----------------------------------------------------------------
    # 1.  Exponents and number of interior zeros (same as in the original code)
    # -----------------------------------------------------------------
    mu = 0.5 * abs(m + s)                # exponent that multiplies (1‑x) near x = +1
    nu = 0.5 * abs(m - s)                # exponent that multiplies (1+x) near x = –1
    N  = int(round(l - max(abs(m), abs(s))))   # interior zeros → phase jump = N*π

    # -----------------------------------------------------------------
    # 2.  Frobenius coefficients a_1 at the two ends (a_θ is forced to 1)
    # -----------------------------------------------------------------

    # NOTE FOR CHRIS: this is where I'm making the implicit linear approximation
    # Not sure if I can even go to quadratic order (since the tolerance will need to be O(ε^3), which means ε > 10^-5)

    def a1_left(lam):
        """
        Series at x = –1.
        The recurrence coefficient α_θ = 2(1+|m‑s|) is **positive**.
        The sign of the c*s term is the “+" one.
        """
        alpha0 = 2.0 * (1.0 + abs(m - s))
        beta0  = (lam - c**2
                 + (abs(m + s) + abs(m - s)) *
                   (abs(m + s) + abs(m - s) + 1.0)
                 + 2.0 * c * s * np.sign(m - s))      # + sign
        a1 = -beta0 / alpha0
        return 1.0, a1                              # a_θ = 1, a_1 as above

    def a1_right(lam):
        """
        Series at x = +1.
        α_θ = 2(1+|m+s|)  (positive)
        The sign of the c*s term is the “‑" one.
        """
        alpha0 = 2.0 * (1.0 + abs(m + s))
        beta0  = (lam - c**2
                 + (abs(m + s) + abs(m - s)) *
                   (abs(m + s) + abs(m - s) + 1.0)
                 - 2.0 * c * s * np.sign(m + s))      # – sign
        a1 = -beta0 / alpha0
        return 1.0, a1


    # -----------------------------------------------------------------
    # 3.  Phase (Prüfer angle) obtained from the two Frobenius series
    # -----------------------------------------------------------------
    def theta_initial(lam):
        # miiiiiight need to add the exponential factors that papers like Leaver (https://arxiv.org/pdf/1408.1860) include
        """
        Returns (θ_left, θ_right) = (θ(w=-epsilon), θ(w=+epsilon)) (BIG number this time!).
        The formula is θ = arctan2(S, p*S') with p = -sech^2 w.
        """
        # ---------- left end (x = –ε) ----------
        x0 = -eps
        p0 = - 1.0 / (np.cosh(x0))**2
        a0_l, a1_l = a1_left(lam)
        neg_tanh_eps = np.tanh(-eps)
        print(neg_tanh_eps)

        S0 = ((1.0 - neg_tanh_eps)**mu) * ((1.0 + neg_tanh_eps)**nu) * (a0_l + a1_l * (1.0 - neg_tanh_eps))

        common_factor0 = ((1.0 - neg_tanh_eps)**mu) * ((1.0 + neg_tanh_eps)**nu)
        dS0 = common_factor0 * (-a1_l * (p0) + (a0_l + a1_l * (1.0 - neg_tanh_eps)) * ((mu / (1.0 + neg_tanh_eps)) + (nu / (1.0 - neg_tanh_eps))))

        th_left = np.arctan2(S0, p0 * dS0)

        # ---------- right end (x = +ε) ----------
        x1 = eps
        p1 = - 1.0 / (np.cosh(x1))**2
        a0_r, a1_r = a1_right(lam)
        tanh_eps = np.tanh(eps)

        S1 = ((1.0 - tanh_eps)**mu) * ((1.0 + tanh_eps)**nu) * (a0_r + a1_r * (1.0 - tanh_eps))

        common_factor1 = ((1.0 - tanh_eps)**mu) * ((1.0 + tanh_eps)**nu)
        dS1 = common_factor1 * (-a1_r * (p1) + (a0_r + a1_r * (1.0 - tanh_eps)) * ((mu / (1.0 + tanh_eps)) + (nu / (1.0 - tanh_eps))))
        
        th_right = np.arctan2(S1, p1 * dS1)

        return th_left, th_right

    def _check_seed():
        """
        Diagnostic: evaluate θ at the two ends for the spherical eigenvalue
        and compare the jump to N*π.
        """
        lam_sph = l * (l + 1) - s * (s + 1)            # exact spherical guess
        thL, thR = theta_initial(lam_sph)

        # bring both angles into the interval (-π,π]  (arctan2 already does this)
        diff = thR - thL # I also think I need to EXPLICITLY add Nπ to this... but that's just a thought
        # not sure if this is also accounting for all the stuff between the actual θ(-1) and θ(+1)?
        # make the difference lie in (0, 2π) for readability
        diff_mod = (diff + 2*np.pi) % (2*np.pi)

        print("=== seed check ===")
        print(f"  λ_spherical = {lam_sph:.12f}")
        print(f"  θ_left  = {thL:.12f}")
        print(f"  θ_right = {thR:.12f}")
        print(f"  raw difference   = {diff:.12f}")
        print(f"  diff modulo 2π   = {diff_mod:.12f}")
        print(f"  Nπ  = {N*np.pi:.12f}")
        print(f"  error (diff‑Nπ) = {diff - N*np.pi:.3e}")
        print("==================\n")

    # call it once at start of the script
    _check_seed()


    # -----------------------------------------------------------------
    # 4.  Prüfer ODE (λ is a parameter)
    # -----------------------------------------------------------------
    # this is likely where I'm expecting some sort of blow up bc I didn't do the tanh w conversion right
    # check appendix K in the overleaf
    def theta_rhs(x, th, lam):
        p = 1.0 - np.tanh(x)**2
        q = (c**2 * np.tanh(x)**2 - 2.0 * c * s * np.tanh(x)
             - (m + s * np.tanh(x))**2 / p + s)
        return (np.cos(th) ** 2) / p + (lam + q) * (np.sin(th) ** 2)

    # -----------------------------------------------------------------
    # 5.  Residual for the meet‑in‑the‑middle condition
    # -----------------------------------------------------------------
    def mid_residual(lam):
        """
        Return θ_left(0) – θ_right(0).
        The root of this function is the eigenvalue.
        """
        print("begin mid_residual")
        th_left_start, th_right_start = theta_initial(lam)

        # integrate from the left hand side up to x = 0
        sol_f = solve_ivp(theta_rhs,
                         t_span=(-eps, 0.0),
                         y0=[th_left_start],
                         args=(lam,),
                         method='RK45',
                         max_step=max_step,
                         t_eval=[0.0])          # force a value at the midpoint

        # integrate backwards from the right hand side down to x = 0
        sol_b = solve_ivp(theta_rhs,
                         t_span=(eps, 0.0),
                         y0=[th_right_start],
                         args=(lam,),
                         method='RK45',
                         max_step=max_step,
                         t_eval=[0.0])

        theta_left_mid  = sol_f.y[0, 0]          # θ(0-)
        theta_right_mid = sol_b.y[0, 0]          # θ(0+)

        return theta_left_mid - theta_right_mid

    # -----------------------------------------------------------------
    # 6.  Bracket the root (exactly as you did before)
    # -----------------------------------------------------------------
    lam0   = l * (l + 1) - s * (s + 1)          # spherical guess
    delta  = max(5.0, 5.0 * abs(c)**2)           # initial half‑width
    lam_lo = lam0 - delta
    lam_hi = lam0 + delta


    a0, a1 = a1_left(lam0)          # lam0 is the spherical guess
    print("a1_left  :", a1)
    a0, a1 = a1_right(lam0)
    print("a1_right :", a1)

    thL, thR = theta_initial(lam0)
    print("θ_R - θ_L (should be Nπ) :", thR - thL, "  Nπ =", N*np.pi)

    for _ in range(30):
        f_lo = mid_residual(lam_lo)
        f_hi = mid_residual(lam_hi)
        print(mid_residual(lam_lo), mid_residual(lam_hi))

        # Guard against integration failures
        if np.isnan(f_lo):
            f_lo = 1e20
        if np.isnan(f_hi):
            f_hi = 1e20

        if f_lo * f_hi < 0:
            break                               # good bracket! yay!
        lam_lo -= delta
        lam_hi += delta
        delta   *= 1.5
    else:
        raise RuntimeError("Failed to bracket the eigenvalue.")

    # -----------------------------------------------------------------
    # 7.  Brent root‑find
    # -----------------------------------------------------------------
    t_start = time.time()
    lam = brentq(mid_residual, lam_lo, lam_hi,
                 xtol=tol, rtol=tol, maxiter=max_iter)
    t_end = time.time()

    return lam, t_end - t_start


# -----------------------------------------------------------------
# Example driver (feel free to change the parameters)
# -----------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: spherical case, s=2, m=2, l=5, c=0  →  λ = 24
    lam, cpu = spheroidal_eigenvalue_mid(s=2.0, m=2.0, l=3.0, c=0.0)
    print(f"λ(s=2,m=2,l=5,c=0) = {lam:.12f}   (cpu = {cpu:.4f} s)")

    # Test 2: spherical case, s=2, m=5, l=10, c=0  →  λ = 80
    #lam2, _ = spheroidal_eigenvalue_mid(s=2.0, m=5.0, l=10.0, c=0.0)
    #print(f"λ(s=2,m=5,l=10,c=0) = {lam2:.12f}")
