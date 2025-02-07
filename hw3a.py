#!/usr/bin/env python3
"""
hw3a.py

This program computes probabilities from a normal distribution either by:
  a) computing the probability given a c-value (using Simpson's 1/3 integration)
  b) computing the corresponding c-value given a target probability using the Secant method.

The program handles both one-sided and two-sided cases.
For one-sided integration, the probability is computed as either P(x < c) or P(x > c).
For two-sided integration, it computes either the central probability, i.e.,
  P(μ – (c–μ) < x < μ + (c–μ)),
or the tail probability (x outside that symmetric interval).

This code uses the GPDF and Probability functions from numericalMethods.
"""

# region imports
from numericalMethods import GPDF, Probability


# endregion

# region helper functions
def compute_probability(c, mean, stDev, OneSided, GT):
    """
    Compute the probability for a normal distribution with given parameters.

    For one-sided integration:
        - if GT==False, returns P(x < c)
        - if GT==True, returns P(x > c)

    For two-sided integration, we interpret c as half-width around the mean:
        - if GT==False, returns the central probability P(μ - (c-μ) < x < μ+(c-μ))
        - if GT==True, returns the tail probability, i.e.,
              1 – P(μ - (c-μ) < x < μ+(c-μ))

    (Note: The actual integration is done by the Probability() function.)
    """
    if OneSided:
        return Probability(GPDF, (mean, stDev), c, GT=GT)
    else:
        # For two-sided, we first compute the one-tail probability
        tail = Probability(GPDF, (mean, stDev), c, GT=True)
        central_prob = 1 - 2 * tail  # P(central interval)
        return central_prob if not GT else 1 - central_prob


def secant_method(func, x0, x1, tol=1e-6, max_iter=100):
    """
    Solve func(x)=0 using the secant method starting with x0 and x1.
    """
    for i in range(max_iter):
        f0 = func(x0)
        f1 = func(x1)
        if abs(f1 - f0) < 1e-12:
            print("Warning: Denominator nearly zero in secant method; returning current estimate.")
            return x1
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x_new - x1) < tol:
            return x_new
        x0, x1 = x1, x_new
    return x_new


# endregion

def main():
    """
    Main interactive loop.
    """
    # Set default parameters:
    mean = 0.0
    stDev = 1.0
    c = 0.5
    OneSided = True  # True: one-sided; False: two-sided symmetric integration
    GT = False  # For one-sided: False means P(x < c), True means P(x > c).
    # For two-sided: False means central probability, True means tail probability.

    yesOptions = ["y", "yes", "true"]

    print("Welcome to hw3a: Normal Distribution Probability Calculator & Inverse Solver")
    print("--------------------------------------------------------------------------")

    again = True
    while again:
        # Ask for distribution parameters:
        resp = input(f"Enter population mean? (default {mean:0.3f}): ").strip()
        if resp != "":
            mean = float(resp)

        resp = input(f"Enter standard deviation? (default {stDev:0.3f}): ").strip()
        if resp != "":
            stDev = float(resp)

        # Ask for integration type:
        resp = input(f"Use one-sided integration? (y/n, default {'y' if OneSided else 'n'}): ").strip().lower()
        if resp:
            OneSided = True if resp in yesOptions else False

        # For one-sided, ask if we want the probability greater than c.
        if OneSided:
            resp = input(f"Compute probability for x > c? (y/n, default {'y' if GT else 'n'}): ").strip().lower()
            if resp:
                GT = True if resp in yesOptions else False
        else:
            # For two-sided, ask if the desired probability is the central (inside) probability.
            resp = input("Should the probability be computed for the central interval? (y/n, default y)\n"
                         "   (Answer 'n' for tail probability outside the symmetric interval): ").strip().lower()
            if resp:
                # If yes, we want the central probability (GT=False); if no, then GT=True.
                GT = False if resp in yesOptions else True
            else:
                GT = False  # default to central probability

        # Ask user which mode they want:
        mode = input(
            "Specify c and compute probability (enter 'c') OR specify probability and solve for c (enter 'p')? (default 'c'): ").strip().lower()

        if mode == "" or mode.startswith("c"):
            # -----------------------
            # Mode 1: Given c, compute probability.
            # -----------------------
            resp = input(f"Enter c value? (default {c:0.3f}): ").strip()
            if resp:
                c = float(resp)
            # Compute probability using our helper function:
            prob = compute_probability(c, mean, stDev, OneSided, GT)

            # Display results:
            if OneSided:
                if GT:
                    print(f"\nResult: P(x > {c:0.3f} | μ = {mean:0.3f}, σ = {stDev:0.3f}) = {prob:0.5f}\n")
                else:
                    print(f"\nResult: P(x < {c:0.3f} | μ = {mean:0.3f}, σ = {stDev:0.3f}) = {prob:0.5f}\n")
            else:
                # For two-sided, interpret c as half-width around the mean.
                lower = mean - (c - mean)
                upper = mean + (c - mean)
                if GT:
                    print(
                        f"\nResult: P(x outside ({lower:0.3f}, {upper:0.3f})) = {prob:0.5f} for N({mean:0.3f}, {stDev:0.3f})\n")
                else:
                    print(
                        f"\nResult: P({lower:0.3f} < x < {upper:0.3f}) = {prob:0.5f} for N({mean:0.3f}, {stDev:0.3f})\n")

        else:
            # -----------------------
            # Mode 2: Given a target probability, solve for c using the secant method.
            # -----------------------
            target_str = input("Enter desired probability: ").strip()
            if target_str:
                target_prob = float(target_str)
            else:
                print("No probability entered; defaulting to 0.5")
                target_prob = 0.5

            # Define our function f(c) = compute_probability(c) - target_prob.
            def f(c_val):
                return compute_probability(c_val, mean, stDev, OneSided, GT) - target_prob

            # Choose two initial guesses for the secant method.
            # The monotonicity depends on the type of integration:
            if not GT:
                # For one-sided P(x < c) or two-sided central probability: f(c) increases with c.
                if OneSided:
                    c0 = mean - 5 * stDev
                    c1 = mean + 5 * stDev
                else:
                    # For two-sided central probability, note that at c=mean, the central interval is degenerate.
                    c0 = mean
                    c1 = mean + 5 * stDev
            else:
                # For one-sided P(x > c) or two-sided tail probability: f(c) decreases with c.
                if OneSided:
                    c0 = mean + 5 * stDev
                    c1 = mean - 5 * stDev
                else:
                    # For two-sided tail probability, at c=mean the tails cover all probability.
                    c0 = mean + 5 * stDev
                    c1 = mean

            # Call the secant method:
            c_solution = secant_method(f, c0, c1)
            # Compute the probability at the solution (should be nearly equal to target_prob)
            prob = compute_probability(c_solution, mean, stDev, OneSided, GT)

            if OneSided:
                if GT:
                    print(
                        f"\nFound c = {c_solution:0.5f} such that P(x > c) ≈ {prob:0.5f} (target {target_prob:0.5f}) for N({mean:0.3f}, {stDev:0.3f})\n")
                else:
                    print(
                        f"\nFound c = {c_solution:0.5f} such that P(x < c) ≈ {prob:0.5f} (target {target_prob:0.5f}) for N({mean:0.3f}, {stDev:0.3f})\n")
            else:
                lower = mean - (c_solution - mean)
                upper = mean + (c_solution - mean)
                if GT:
                    print(
                        f"\nFound c = {c_solution:0.5f} such that P(x outside ({lower:0.3f}, {upper:0.3f})) ≈ {prob:0.5f} (target {target_prob:0.5f}) for N({mean:0.3f}, {stDev:0.3f})\n")
                else:
                    print(
                        f"\nFound c = {c_solution:0.5f} such that P({lower:0.3f} < x < {upper:0.3f}) ≈ {prob:0.5f} (target {target_prob:0.5f}) for N({mean:0.3f}, {stDev:0.3f})\n")

        # Ask if the user wants to run another calculation:
        resp = input("Run again? (y/n): ").strip().lower()
        if resp not in yesOptions:
            again = False


if __name__ == "__main__":
    main()
