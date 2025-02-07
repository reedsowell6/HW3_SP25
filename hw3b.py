#!/usr/bin/env python3
"""
hw3b.py

This program computes the right-hand side of the equation for the t-distribution:

   F(z) = K_m * ∫₋∞ᶻ (1 + u²/m)^(-(m+1)/2) du,
   where K_m = Γ((m+1)/2) / (sqrt(mπ) * Γ(m/2)).

For numerical reasons, we use the symmetry of the t-distribution to write:
  For z ≥ 0:
      F(z) = 0.5 + K_m * ∫₀ᶻ (1 + u²/m)^(-(m+1)/2) du,
  For z < 0:
      F(z) = 0.5 - K_m * ∫₀|ᶻ| (1 + u²/m)^(-(m+1)/2) du.

The Gamma function is defined by
    Γ(α) = ∫₀∞ e^(-t) t^(α-1) dt,
but we use Python's math.gamma for convenience.

The program prompts the user to input the degrees of freedom (m) and a value of z,
and then prints the computed probability F(z).
"""

import math


# --------------------------------------------------------------------
# Define the integrand for the t-distribution (without the normalizing constant K_m).
# --------------------------------------------------------------------
def t_integrand(u, m):
    """
    Computes (1 + u^2/m)^(-(m+1)/2) for a given u and degrees of freedom m.
    """
    return (1 + (u * u) / m) ** (-(m + 1) / 2)


# --------------------------------------------------------------------
# Simpson's Rule Integration
# --------------------------------------------------------------------
def simpson_rule(func, a, b, m, n=1000):
    """
    Approximate the integral of func(u, m) from u=a to u=b using Simpson's rule.

    Parameters:
      func : function accepting (u, m)
      a, b : integration limits
      m    : degrees of freedom passed to func
      n    : number of subintervals (n must be even; if odd, it is incremented by 1)

    Returns:
      Approximation of the integral.
    """
    if n % 2 == 1:
        n += 1  # ensure n is even
    h = (b - a) / n
    s = func(a, m) + func(b, m)
    for i in range(1, n):
        u = a + i * h
        if i % 2 == 0:
            s += 2 * func(u, m)
        else:
            s += 4 * func(u, m)
    return s * h / 3


# --------------------------------------------------------------------
# Compute the t-distribution CDF using symmetry.
# --------------------------------------------------------------------
def compute_t_cdf(z, m):
    """
    Computes the cumulative probability F(z) for the t-distribution with
    degrees of freedom m.

    Uses the normalization constant:
      K_m = Γ((m+1)/2) / (sqrt(mπ)*Γ(m/2))
    and the symmetry property:
      For z >= 0: F(z) = 0.5 + K_m ∫₀ᶻ (1 + u²/m)^(-(m+1)/2) du,
      For z < 0:  F(z) = 0.5 - K_m ∫₀|ᶻ| (1 + u²/m)^(-(m+1)/2) du.
    """
    # Compute K_m using math.gamma
    Km = math.gamma((m + 1) / 2) / (math.sqrt(m * math.pi) * math.gamma(m / 2))

    if z == 0:
        return 0.5
    elif z > 0:
        integral = simpson_rule(t_integrand, 0, z, m, n=1000)
        return 0.5 + Km * integral
    else:
        integral = simpson_rule(t_integrand, 0, -z, m, n=1000)
        return 0.5 - Km * integral


# --------------------------------------------------------------------
# Main interactive loop
# --------------------------------------------------------------------
def main():
    print("t-Distribution CDF Calculator")
    print("Computes F(z) = K_m ∫₋∞ᶻ (1 + u²/m)^(-(m+1)/2) du")
    print("with K_m = Γ((m+1)/2)/(√(mπ)*Γ(m/2))")
    print("-----------------------------------------------------")

    while True:
        try:
            m_input = input("Enter degrees of freedom (m) (e.g., 7, 11, or 15): ").strip()
            m = float(m_input)
            if m <= 0:
                print("Degrees of freedom must be positive. Please try again.")
                continue

            z_input = input("Enter z value: ").strip()
            z_val = float(z_input)
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            continue

        # Compute the cumulative probability F(z)
        probability = compute_t_cdf(z_val, m)
        print(f"\nFor m = {m:.0f} and z = {z_val:.4f}, the computed probability F(z) = {probability:.5f}\n")

        # Ask if the user wishes to perform another calculation
        again = input("Compute another value? (y/n): ").strip().lower()
        if again not in ['y', 'yes']:
            break


if __name__ == "__main__":
    main()
