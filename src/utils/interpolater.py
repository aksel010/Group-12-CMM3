"""
Interpolation and comparison utilities for 1D data.

Implements:
- Custom cubic spline interpolation and its derivative.
- SciPy cubic spline (interp1d) usage and finite-difference derivative.
- Newton divided difference interpolation.
- Comparison of interpolation methods and their accuracy.

Main Functions:
    - cubic_spline_interpolation
    - cubic_spline_derivative
    - scipy_cubic_interpolation
    - scipy_cubic_derivative
    - newton_divided_difference
    - compare_interpolation_methods
    - compare_interpolation_accuracy

"""
import numpy as np
from scipy.interpolate import interp1d

# (Functions unchanged - only docstrings, annotation, and comments augmented)

def cubic_spline_interpolation(x_data, y_data, x_query):
    """
    Perform cubic spline interpolation at given query points.

    Args:
        x_data (array-like): x-coordinates of data points.
        y_data (array-like): y-coordinates of data points.
        x_query (float or array-like): x value(s) to interpolate at.

    Returns:
        float or array: Interpolated y value(s).
    """
    #... (original function body unchanged - see repository for details)
    # [Full function body retained]
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data) - 1
    H = [x_data[i+1] - x_data[i] for i in range(n)]
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    for i in range(1, n):
        A[i, i] = 2 * (H[i-1] + H[i])
    for i in range(1, n):
        A[i, i+1] = H[i]
    for i in range(1, n):
        A[i, i-1] = H[i-1]
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1, n):
        b[i] = 6 * ((y_data[i+1] - y_data[i]) / H[i] - (y_data[i] - y_data[i-1]) / H[i-1])
    M = np.linalg.solve(A, b)
    coefficients = []
    for i in range(n):
        a = y_data[i]
        b_coeff = (y_data[i+1] - y_data[i]) / H[i] - H[i] * (2 * M[i] + M[i+1]) / 6
        c = M[i] / 2
        d = (M[i+1] - M[i]) / (6 * H[i])
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
    Calculate the first derivative of cubic spline at given x value(s).

    Args:
        x_data (array-like): x-coordinates of data points.
        y_data (array-like): y-coordinates of data points.
        x_query (float or array-like): x value(s) for the derivative.

    Returns:
        float or array: First derivative value(s).
    """
    #...(body unchanged)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data) - 1
    H = [x_data[i+1] - x_data[i] for i in range(n)]
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    for i in range(1, n):
        A[i, i] = 2 * (H[i-1] + H[i])
    for i in range(1, n):
        A[i, i+1] = H[i]
    for i in range(1, n):
        A[i, i-1] = H[i-1]
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1, n):
        b[i] = 6 * ((y_data[i+1] - y_data[i]) / H[i] - (y_data[i] - y_data[i-1]) / H[i-1])
    M = np.linalg.solve(A, b)
    coefficients = []
    for i in range(n):
        a = y_data[i]
        b_coeff = (y_data[i+1] - y_data[i]) / H[i] - H[i] * (2 * M[i] + M[i+1]) / 6
        c = M[i] / 2
        d = (M[i+1] - M[i]) / (6 * H[i])
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
    Use SciPy cubic spline interpolation for given query points.

    Args:
        x_data (array-like): x-coordinates of data points.
        y_data (array-like): y-coordinates of data points.
        x_query (float or array-like): x value(s) to interpolate at.

    Returns:
        float or array: Interpolated y value(s).
    """
    cubic_spline = interp1d(x_data, y_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return cubic_spline(x_query)

def scipy_cubic_derivative(x_data, y_data, x_query, h=1e-6):
    """
    Calculate the finite-difference derivative of the SciPy cubic spline.

    Args:
        x_data (array-like): x-coordinates of data points.
        y_data (array-like): y-coordinates of data points.
        x_query (float or array-like): Point(s) for derivative.
        h (float): Small step for finite-difference.

    Returns:
        float or array: Derivative value(s) of the spline at x_query.
    """
    cubic_spline = interp1d(x_data, y_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
    if hasattr(x_query, '__iter__'):
        derivatives = []
        for xq in x_query:
            f_plus = cubic_spline(xq + h)
            f_minus = cubic_spline(xq - h)
            derivative = (f_plus - f_minus) / (2 * h)
            derivatives.append(derivative)
        return np.array(derivatives)
    else:
        f_plus = cubic_spline(x_query + h)
        f_minus = cubic_spline(x_query - h)
        return (f_plus - f_minus) / (2 * h)

def newton_divided_difference(x_data, y_data, x_query):
    """
    Newton divided difference interpolation at query points.

    Args:
        x_data (array-like): x-coordinates of data points.
        y_data (array-like): y-coordinates of data points.
        x_query (float or array-like): x value(s) to interpolate at.

    Returns:
        float or array: Interpolated y value(s).
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
    Compare interpolation methods: 'spline', 'scipy_cubic', or 'newton'.

    Args:
        x_data (array-like): x-coordinates of data points.
        y_data (array-like): y-coordinates of data points.
        x_query (float or array-like): Query locations.
        method (str): Method name.

    Returns:
        float or array: Interpolated y value(s).
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
    Compare all interpolation methods for accuracy at test points.

    Args:
        x_data (array-like): x-coordinates of data points.
        y_data (array-like): y-coordinates of data points.
        test_points (array-like, optional): Points to evaluate; defaults to linspace in data range.

    Returns:
        dict: Results, RMSE errors, interpolated values.
    """
    if test_points is None:
        test_points = np.linspace(min(x_data), max(x_data), 100)
    spline_values = compare_interpolation_methods(x_data, y_data, test_points, 'spline')
    scipy_values = compare_interpolation_methods(x_data, y_data, test_points, 'scipy_cubic')
    newton_values = compare_interpolation_methods(x_data, y_data, test_points, 'newton')
    true_values = spline_values
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
