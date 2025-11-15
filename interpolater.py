def interpolate(x,y)
    
# Cubic Spline Interpolation
    def cubic_spline_coefficients(x_data, y_data):
        global SPLINE_COEFFICIENTS, I_ARRAY
        n = len(x_data) - 1
        I_ARRAY = np.array(x_data)

        # Step 1: Calculate h (interval widths)
        H = [x_data[i+1] - x_data[i] for i in range(n)]
        
        # Step 2: Set up tridiagonal system for second derivatives
        A = np.zeros((n+1, n+1))
        b = np.zeros(n+1)
        
        # Main diagonal
        for i in range(1, n):
            A[i, i] = 2 * (H[i-1] + H[i])
        
        # Upper diagonal
        for i in range(1, n):
            A[i, i+1] = H[i]
        
        # Lower diagonal 
        for i in range(1, n):
            A[i, i-1] = H[i-1]
        
        # Boundary conditions (natural spline)
        A[0, 0] = 1
        A[n, n] = 1
        
        # Right-hand side
        for i in range(1, n):
            b[i] = 6 * ((y_data[i+1] - y_data[i]) / H[i] - (y_data[i] - y_data[i-1]) / H[i-1])
        
        # Step 3: Solve for second derivatives M
        M = np.linalg.solve(A, b)
        
        # Step 4: Calculate coefficients for each segment
        coefficients = []
        SPLINE_COEFFICIENTS = coefficients
        for i in range(n):
            a = y_data[i]
            b = (y_data[i+1] - y_data[i]) / H[i] - H[i] * (2 * M[i] + M[i+1]) / 6
            c = M[i] / 2
            d = (M[i+1] - M[i]) / (6 * H[i])
            coefficients.append((a, b, c, d, x_data[i], x_data[i+1]))
        
        return coefficients

    def cubic_spline_interpolation(x_data, y_data, x_query):
        # Ensure coefficients are calculated before interpolating
        if SPLINE_COEFFICIENTS is None:
            cubic_spline_coefficients(x_data, y_data)
        
        coefficients = SPLINE_COEFFICIENTS
        
        # Find the correct segment
        for coeff in coefficients:
            a, b, c, d, x_start, x_end = coeff
            if x_start <= x_query <= x_end + 1e-9:
                dx = x_query - x_start
                return a + b * dx + c * dx**2 + d * dx**3
        
        # Extrapolation handling
        if x_query < x_data[0]:
            coeff = coefficients[0]
            a, b, c, d, x_start, x_end = coeff
            dx = x_query - x_start
            return a + b * dx + c * dx**2 + d * dx**3
        else:
            coeff = coefficients[-1]
            a, b, c, d, x_start, x_end = coeff
            dx = x_query - x_start
            return a + b * dx + c * dx**2 + d * dx**3
    