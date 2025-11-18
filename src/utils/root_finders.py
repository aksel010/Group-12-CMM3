"""
Root-finding algorithms: Newton-Raphson and Bisection methods.

This module provides basic root-solving strategies with clear PEP8-compliant docstrings and function annotations.
"""

def newton(f, Df, x0, epsilon, max_iter, args=()):
    """
    Newton-Raphson method for finding roots of a function.

    Args:
        f (callable): Function whose root is sought.
        Df (callable): Derivative of f.
        x0 (float): Initial guess.
        epsilon (float): Convergence threshold for |f(x)|.
        max_iter (int): Maximum allowed iterations.
        args (tuple, optional): Arguments for f and Df.

    Returns:
        float or None: Approximated root, or None if not found.
    """
    xn = x0
    for n in range(max_iter):
        fxn = f(xn, *args) if args else f(xn)
        Dfxn = Df(xn, *args) if args else Df(xn)
        if abs(fxn) < epsilon:
            return xn
        if Dfxn == 0:
            print("Zero derivative. No solution found.")
            return None
        xn = xn - fxn / Dfxn
        if xn < 0:
            xn = 1e-6
    print("Exceeded maximum iterations. No solution found.")
    return None

def bisection(f, a, b, tolerance, max_iter=100):
    """
    Bisection method for finding function roots in an interval.

    Args:
        f (callable): Function whose root is sought.
        a (float): Left endpoint of initial interval.
        b (float): Right endpoint of initial interval.
        tolerance (float): Threshold for |f(c)| or interval width.
        max_iter (int, optional): Max allowed iterations. Default is 100.

    Returns:
        float: Approximated root location.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Bisection method requires f(a) and f(b) to have opposite signs.")
    a_i, b_i = a, b
    for _ in range(max_iter):
        c_i = (a_i + b_i) / 2
        f_c = f(c_i)
        if abs(f_c) < tolerance or (b_i - a_i) / 2 < tolerance:
            return c_i
        f_a = f(a_i)
        if f_a * f_c < 0:
            b_i = c_i
        else:
            a_i = c_i
    return (a_i + b_i) / 2
