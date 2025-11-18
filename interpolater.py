import numpy as np
from scipy.interpolate import interp1d

def cubic_spline_interpolation(x_data, y_data, x_query):
    """
    Perform cubic spline interpolation for provided data points and query.
    Args:
        x_data (array-like): Input x-coordinates
        y_data (array-like): Input y-coordinates
        x_query (float or array-like): x values to interpolate
    Returns:
        float or np.ndarray: Interpolated y value(s) at x_query
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data) - 1
    h_values = [x_data[i+1] - x_data[i] for i in range(n)]
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    for i in range(1, n):
        A[i, i] = 2 * (h_values[i-1] + h_values[i])
        A[i, i+1] = h_values[i]
        A[i, i-1] = h_values[i-1]
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1, n):
        b[i] = 6 * ((y_data[i+1] - y_data[i]) / h_values[i] - (y_data[i] - y_data[i-1]) / h_values[i-1])
    M = np.linalg.solve(A, b)
    coefficients = []
    for i in range(n):
        a = y_data[i]
        b_coeff = (y_data[i+1] - y_data[i]) / h_values[i] - h_values[i] * (2 * M[i] + M[i+1]) / 6
        c = M[i] / 2
        d = (M[i+1] - M[i]) / (6 * h_values[i])
        coefficients.append((a, b_coeff, c, d, x_data[i], x_data[i+1]))
    if hasattr(x_query, '__iter__'):
        results = []
        for xq in x_query:
            interpolated = None
            for coeff in coefficients:
                a, b, c, d, x_start, x_end = coeff
                if x_start <= xq <= x_end + 1e-9:
                    dx = xq - x_start
                    results.append(a + b * dx + c * dx**2 + d * dx**3)
                    interpolated = True
                    break
            if not interpolated:
                if xq < x_data[0]:
                    coeff = coefficients[0]
                else:
                    coeff = coefficients[-1]
                a, b, c, d, x_start, x_end = coeff
                dx = xq - x_start
                results.append(a + b * dx + c * dx**2 + d * dx**3)
        return np.array(results)
    else:
        for coeff in coefficients:
            a, b, c, d, x_start, x_end = coeff
            if x_start <= x_query <= x_end + 1e-9:
                dx = x_query - x_start
                return a + b * dx + c * dx**2 + d * dx**3
        if x_query < x_data[0]:
            coeff = coefficients[0]
        else:
            coeff = coefficients[-1]
        a, b, c, d, x_start, x_end = coeff
        dx = x_query - x_start
        return a + b * dx + c * dx**2 + d * dx**3

def cubic_spline_derivative(x_data, y_data, x_query):
    """
    Compute the first derivative of cubic spline interpolation at the query.
    Args:
        x_data (array-like): Input x-coordinates
        y_data (array-like): Input y-coordinates
        x_query (float or array-like): x values to calculate derivative
    Returns:
        float or np.ndarray: Derivative value(s) at x_query
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data) - 1
    h_values = [x_data[i+1] - x_data[i] for i in range(n)]
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    for i in range(1, n):
        A[i, i] = 2 * (h_values[i-1] + h_values[i])
        A[i, i+1] = h_values[i]
        A[i, i-1] = h_values[i-1]
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1, n):
        b[i] = 6 * ((y_data[i+1] - y_data[i]) / h_values[i] - (y_data[i] - y_data[i-1]) / h_values[i-1])
    M = np.linalg.solve(A, b)
    coefficients = []
    for i in range(n):
        a = y_data[i]
        b_coeff = (y_data[i+1] - y_data[i]) / h_values[i] - h_values[i] * (2 * M[i] + M[i+1]) / 6
        c = M[i] / 2
        d = (M[i+1] - M[i]) / (6 * h_values[i])
        coefficients.append((a, b_coeff, c, d, x_data[i], x_data[i+1]))
    if hasattr(x_query, '__iter__'):
        results = []
        for xq in x_query:
            derivative_found = None
            for coeff in coefficients:
                a, b, c, d, x_start, x_end = coeff
                if x_start <= xq <= x_end + 1e-9:
                    dx = xq - x_start
                    results.append(b + 2 * c * dx + 3 * d * dx**2)
                    derivative_found = True
                    break
            if not derivative_found:
                if xq < x_data[0]:
                    coeff = coefficients[0]
                else:
                    coeff = coefficients[-1]
                a, b, c, d, x_start, x_end = coeff
                dx = xq - x_start
                results.append(b + 2 * c * dx + 3 * d * dx**2)
        return np.array(results)
    else:
        for coeff in coefficients:
            a, b, c, d, x_start, x_end = coeff
            if x_start <= x_query <= x_end + 1e-9:
                dx = x_query - x_start
                return b + 2 * c * dx + 3 * d * dx**2
        if x_query < x_data[0]:
            coeff = coefficients[0]
        else:
            coeff = coefficients[-1]
        a, b, c, d, x_start, x_end = coeff
        dx = x_query - x_start
        return b + 2 * c * dx + 3 * d * dx**2

def scipy_cubic_interpolation(x_data, y_data, x_query):
    """
    Interpolate using SciPy's cubic spline.
    """
    cubic_spline = interp1d(x_data, y_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return cubic_spline(x_query)

def scipy_cubic_derivative(x_data, y_data, x_query, h=1e-6):
    """
    Estimate cubic spline derivative by finite differences.
    """
    cubic_spline = interp1d(x_data, y_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
    if hasattr(x_query, '__iter__'):
        derivatives = []
        for xq in x_query:
            f_plus = cubic_spline(xq + h)
            f_minus = cubic_spline(xq - h)
            derivatives.append((f_plus - f_minus) / (2 * h))
        return np.array(derivatives)
    else:
        f_plus = cubic_spline(x_query + h)
        f_minus = cubic_spline(x_query - h)
        return (f_plus - f_minus) / (2 * h)

def newton_divided_difference(x_data, y_data, x_query):
    """
    Newton divided difference interpolation implementation.
    Args:
        x_data (array-like)
        y_data (array-like)
        x_query (float or array-like)
    Returns:
        float or np.ndarray: Interpolated y value(s)
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data)
    F = np.zeros((n, n))
    F[:, 0] = y_data
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i+1, j-1] - F[i, j-1]) / (x_data[i+j] - x_data[i])
    def interpolate_single(xq):
        result = F[0, 0]
        product_term = 1.0
        for i in range(1, n):
            product_term *= (xq - x_data[i-1])
            result += F[0, i] * product_term
        return result
    if hasattr(x_query, '__iter__'):
        return np.array([interpolate_single(xq) for xq in x_query])
    else:
        return interpolate_single(x_query)

def compare_interpolation_methods(x_data, y_data, x_query, method='spline'):
    """
    Compare interpolation (custom spline, scipy, newton).
    """
    if method.lower() == 'spline':
        return cubic_spline_interpolation(x_data, y_data, x_query)
    elif method.lower() == 'scipy_cubic':
        return scipy_cubic_interpolation(x_data, y_data, x_query)
    elif method.lower() == 'newton':
        return newton_divided_difference(x_data, y_data, x_query)
    else:
        raise ValueError("Method must be 'spline', 'scipy_cubic', or 'newton'")

def compare_interpolation_accuracy(x_data, y_data, test_points=None):
    """
    Compare accuracy of interpolation methods using RMSE.
    """
    if test_points is None:
        test_points = np.linspace(min(x_data), max(x_data), 100)
    spline_values = compare_interpolation_methods(x_data, y_data, test_points, 'spline')
    scipy_values = compare_interpolation_methods(x_data, y_data, test_points, 'scipy_cubic')
    newton_values = compare_interpolation_methods(x_data, y_data, test_points, 'newton')
    true_values = compare_interpolation_methods(x_data, y_data, test_points, 'spline')
    spline_errors = spline_values - true_values
    scipy_errors = scipy_values - true_values
    newton_errors = newton_values - true_values
    spline_rmse = np.sqrt(np.mean(spline_errors**2))
    scipy_rmse = np.sqrt(np.mean(scipy_errors**2))
    newton_rmse = np.sqrt(np.mean(newton_errors**2))
    return {
        'test_points': test_points,
        'spline_values': spline_values,
        'scipy_values': scipy_values,
        'newton_values': newton_values,
        'true_values': true_values,
        'spline_rmse': spline_rmse,
        'scipy_rmse': scipy_rmse,
        'newton_rmse': newton_rmse,
        'spline_errors': spline_errors,
        'scipy_errors': scipy_errors,
        'newton_errors': newton_errors
    }
