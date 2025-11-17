def newton(f, Df, x0, epsilon, max_iter, args=()):
    """
    Newton-Raphson method for finding roots.
    """
    xn = x0
    for n in range(0, max_iter):
        # Pass additional arguments to the functions
        if args:
            fxn = f(xn, *args)
            Dfxn = Df(xn, *args)
        else:
            fxn = f(xn)
            Dfxn = Df(xn)
            
        if abs(fxn) < epsilon:
            return xn
            
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
            
        xn = xn - fxn / Dfxn
        
        # Ensure positive mass flow rate
        if xn < 0:
            xn = 1e-6  # Small positive value
            
    print('Exceeded maximum iterations. No solution found.')
    return None

def bisection(f, a, b, tolerance, max_iter=100):

    if f(a) * f(b) >= 0:
        raise ValueError("Bisection method requires f(a) and f(b) to have opposite signs.")
    
    a_i = a
    b_i = b
    
    for _ in range(max_iter):
        c_i = (a_i + b_i) / 2
        f_c = f(c_i)
        
        if abs(f_c) < tolerance or (b_i - a_i) / 2 < tolerance:
            return c_i
            
        f_a = f(a_i) # Re-evaluate f(a_i)
        
        if f_a * f_c < 0:
            b_i = c_i
        else:
            a_i = c_i
            
    return (a_i + b_i) / 2 # Return best estimate after max_iter