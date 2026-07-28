# --------------------------------------------------------------
#  spheroidal eigenvalue via Prüfer (phase) shooting
# --------------------------------------------------------------
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import time


def spheroidal_eigenvalue(s, m, l, c, eps=1e-8, max_iter=100, tol=1e-14):
    """
    Return the spin‑weighted spheroidal eigenvalue lambda_{slm}(c)
    for the given parameters by solving the Prüfer equation.
    """
    # Note: tolerance is related to eps**(terms in Leaver series). 
    # So (10^-6)**(2 (linear)) = 10^-12 tolerance or above
    # ----------------------------------------------------------
    # 1.  Exponents at the two ends and number of interior zeros
    # ----------------------------------------------------------
    twice_mu = abs(m + s)
    twice_nu = abs(m - s)
    mu = 0.5 * twice_mu     # exponent approaching x = +1
    nu = 0.5 * twice_nu     # exponent approaching x = -1
    N  = int(round(l - max(abs(m), abs(s))))   # interior nodes (like a standing wave!)

    # ----------------------------------------------------------
    # 2.  Coefficients a0 and a1 of the Frobenius series at x = -1
    # ----------------------------------------------------------
    def a1_coefficient(lam):
        """return a1 = -beta0(lambda)/alpha0. a0 is set to 1."""

        # alpha0 = 2.0 * (1.0) * (1.0 + abs(m + s)) 
        # beta0 = -2 * c * (abs(m - s) + s + 1) + (abs(m + s) + abs(m - s)) * (abs(m + s) + abs(m - s) + 1) - c**2 - s * (s + 1) - lam

        a0 = 1.0
        alpha0 = 2.0 * (1.0) * (1.0 + twice_mu) 
        beta0 = -2 * c * (twice_nu + s + 1) + (twice_mu + twice_nu) * (twice_mu + twice_nu + 1) - c**2 - s * (s + 1) - lam 
        a1 = -beta0 / alpha0
        return a0, a1

    # ----------------------------------------------------------
    # 3.  Initial angle theta(-1+epsilon) obtained from the series
    # ----------------------------------------------------------
    def theta_initial(lam):
        """theta_0 = arctan[S/(pS')] evaluated at x = -1+epsilon."""
        a0, a1 = a1_coefficient(lam)

        # Three-term recurrence math to get S/pS' (see overleaf)
        denom_frac = nu - (mu * eps)/(2 - eps)
        numerator = 0.5 * (a0 + a1 * eps)
        denominator = (denom_frac) * (a0 + a1 * eps) + a1 * eps

        # theta_0 from the Prüfer definition
        th0 = np.arctan2(numerator, denominator) # arctan2 gives the correct quadrant (https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html)
        return th0

    # ----------------------------------------------------------
    # 4.  Right‑hand side of the Prüfer ODE – lambda is a parameter
    # ----------------------------------------------------------
    def theta_rhs(x, th, lam):
        p = 1.0 - x**2
        # q(x) from the original equation, including the constant +s
        q = (c**2 * x**2 - 2.0 * c * s * x - (m + s * x)**2 / p + s)
        return (np.cos(th)**2) / p + (lam + q) * (np.sin(th)**2)

    # ----------------------------------------------------------
    # 5.  Phase‑jump function  delta_theta(lambda) – Npi
    # ----------------------------------------------------------
    def phase_jump(lam):
        phase_jump_time_start = time.time()
        th0 = theta_initial(lam)

        sol = solve_ivp(theta_rhs,
                        t_span=(-1.0 + eps, 1.0 - eps),
                        y0=[th0],
                        args=(lam,),
                        method='DOP853', # high‑order, stiff‑friendly (try some others? Calling the function over and over againa t each timestep....)
                        # Make a step size that is small near the poles and grows toward the middle? Or a uniform step size with a transformed indep. variable?
                        rtol=1e-12, atol=1e-14,
                        max_step=0.01)     # keep steps reasonable near the poles

        th1 = sol.y[0, -1]
        phase_jump_time_end = time.time()
        print(phase_jump_time_start - phase_jump_time_end)
        return th1 - th0 - N * np.pi

    # ----------------------------------------------------------
    # 6.  Bracket the root
    # ----------------------------------------------------------
    # The spherical value lambda_0 = l(l+1) - s(s+1) is a very good centre. # THIS IS 
    lam0 = l * (l + 1) - s * (s + 1)
    # A conservative interval: +/- (5 + 5|c|^2) around the spherical value.
    delta = max(5.0, 5.0 * abs(c)**2)

    lam_low  = lam0 - delta
    lam_high = lam0 + delta

    # Enlarge until we see a sign change (the function is monotone)
    for _ in range(30):
        print("expansion") # this is to check computation time
        f_low  = phase_jump(lam_low)
        f_high = phase_jump(lam_high)
        if np.isnan(f_low) or np.isnan(f_high):
            # If the integrator blew up, treat the value as huge. 
            # (maybe pick a smaller value, so if we DO use Riccati equation, it blows up slower?)
            f_low  = np.sign(f_low)  * 1e20 if np.isnan(f_low)  else f_low
            f_high = np.sign(f_high) * 1e20 if np.isnan(f_high) else f_high

        if f_low * f_high < 0:
            break # Good bracket found
        # Otherwise stretch the interval
        lam_low  -= delta
        lam_high += delta
        delta *= 1.5 # Expand progressively

    else:
        raise RuntimeError("Failed to bracket the eigenvalue.")

    # ----------------------------------------------------------
    # 7.  Root‑find with Brent’s method (guaranteed convergence) (I hope)
    # ----------------------------------------------------------
    t_start = time.time()
    lam = brentq(phase_jump, lam_low, lam_high,
                 xtol=tol, rtol=tol, maxiter=max_iter)
    t_end = time.time()

    print(f"lam time: = {t_end-t_start}")

    return lam, t_end-t_start
# -----------------------------------------------------------------
# Example usage 
# -----------------------------------------------------------------
if __name__ == "__main__":

    t0 = time.time()
    # Spherical case, s=2, m=2, l=5, c=0
    lam, time_1 = spheroidal_eigenvalue(s=2.0, m=15.0, l=20.0, c=1.0)

    t1 = time.time()
    # A non‑zero oblateness parameter
    #lam2, time_2 = spheroidal_eigenvalue(s=2.0, m=50.0, l=100.0, c=0.0001)

    t2 = time.time()
    # Big oblateness parameter
    #lam3, time_3 = spheroidal_eigenvalue(s=2.0, m=50.0, l=100.0, c=0.01)

    # Print the values
    t3 = time.time() # EXCURSIONS IN C SPACE!!! MORE INTERATIONS??
    print(f"lambda(s=2,m=15,l=20,c=1.0,max_iter=100,tol=1e-14) = {lam:.12f}, time = {t1-t0}, brent_time = {time_1}")
    #print(f"lambda(s=2,m=50,l=100,c=0.0001,max_iter=500) = {lam2:.12f}, time = {t2-t1}, brent_time = {time_2}")
    #print(f"lambda(s=2,m=50,l=100,c=0.001,max_iter=500) = {lam3:.12f}, time = {t3-t2}, brent_time = {time_3}")

    # Put time it in, and output these to a text file!!!
    # https://docs.python.org/3/library/timeit.html
    # Woudl SWSH mathematica package be a good check?
    # Or spectral method... or Leaver's in general... lol

    # PUT THIS ON THE GITHUB!!!!!! Subfolder for the project (PRUF OF CONCEPT!!!)
    # Functions inside functions the problem?? Am I redefining brent's method?


    # SOLVE INDEPENDENTLY!! THATS THE WHOLE POINT!!!!