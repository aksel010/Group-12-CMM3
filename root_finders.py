"""
Root-finding methods: Newton-Raphson and Bisection
"""

def newton(f, Df, x0, epsilon, max_iter, args=()):
    """
    Newton-Raphson method for finding roots of a function.

    Parameters
    ----------
    f : callable
        Function whose root is sought.
    Df : callable
        Derivative of the function f.
    x0 : float
        Initial guess.
    epsilon : float
        Convergence tolerance for |f(x)|.
    max_iter : int
        Maximum number of iterations.
    args : tuple, optional
        Additional arguments to pass to f and Df.

    Returns
    -------
    float or None
        Approximated root or None if convergence failed.
    """
    xn = x0
    for n in range(max_iter):
        # Evaluate function and derivative
        fxn = f(xn, *args) if args else f(xn)
        Dfxn = Df(xn, *args) if args else Df(xn)

        # Convergence check
        if abs(fxn) < epsilon:
            return xn

        # Check for zero derivative
        if Dfxn == 0:
            print("Zero derivative. No solution found.")
            return None

        # Newton-Raphson step
        xn = xn - fxn / Dfxn

        # Ensure positive solution if required (e.g., mass flow rate)
        if xn < 0:
            xn = 1e-6

    print("Exceeded maximum iterations. No solution found.")
    return None


def bisection(f, a, b, tolerance, max_iter=100):
    """
    Bisection method for finding roots of a function.

    Parameters
    ----------
    f : callable
        Function whose root is sought.
    a : float
        Left endpoint of initial interval.
    b : float
        Right endpoint of initial interval.
    tolerance : float
        Convergence criterion for |f(c)| or interval width.
    max_iter : int, optional
        Maximum number of iterations (default: 100).

    Returns
    -------
    float
        Approximated root.
    """
    if f(a) * f(b) >= 0:
        raise ValueError(
            "Bisection method requires f(a) and f(b) to have opposite signs."
        )

    a_i, b_i = a, b

    for _ in range(max_iter):
        c_i = (a_i + b_i) / 2
        f_c = f(c_i)

        # Check convergence
        if abs(f_c) < tolerance or (b_i - a_i) / 2 < tolerance:
            return c_i

        f_a = f(a_i)  # Re-evaluate f(a_i)

        # Update interval
        if f_a * f_c < 0:
            b_i = c_i
        else:
            a_i = c_i

    # Return best estimate if max iterations reached
    return (a_i + b_i) / 2
