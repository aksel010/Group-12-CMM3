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
            print('Found solution after', n, 'iterations.')
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