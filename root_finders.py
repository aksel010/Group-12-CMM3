"""
root_finders.py
Root-finding: Newton-Raphson method and Bisection method (PEP8 & docstring standard).
"""

def newton_method(f, df, x0, epsilon, max_iter, args=()):
    """
    Newton-Raphson method for finding roots of a function.
    Args:
        f (callable): Function whose root is sought.
        df (callable): Derivative of the function.
        x0 (float): Initial guess for the root.
        epsilon (float): Convergence tolerance for |f(x)|.
        max_iter (int): Maximum iterations allowed.
        args (tuple): Additional parameters for f and df.
    Returns:
        float or None: Approximated root (None if fails).
    """
    xn = x0
    for n in range(max_iter):
        fxn = f(xn, *args) if args else f(xn)
        Dfxn = df(xn, *args) if args else df(xn)
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

def bisection_method(f, a, b, tolerance, max_iter=100):
    """
    Bisection method for finding roots of a function.
    Args:
        f (callable): Function whose root is sought.
        a (float): Left endpoint.
        b (float): Right endpoint.
        tolerance (float): Convergence tolerance.
        max_iter (int): Maximum iterations allowed.
    Returns:
        float: Approximated root.
    Raises:
        ValueError: if f(a) and f(b) do not have opposite signs.
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
