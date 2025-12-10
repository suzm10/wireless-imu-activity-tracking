from scipy.optimize import curve_fit
import numpy as np

# Straightforard: get the corresponding y-value for some x-value in a polynomial function
def polynomial_func(reading, a, b, c):
    return a * reading ** 2 + b * reading + c

# With known baseline data, retrieve the optimal parameters for a fitted polynomial function
# that can be used inside of the Feather's microcontroller code
def get_estimate(number):
    # Initializing baseline data
    baseline_weights = np.array([0, 2.5, 5, 10, 25])
    baseline_readings = np.array([400, 480, 520, 580, 675])

    # Get the parameters for the best-fitted polynomial function
    params, _ = curve_fit(polynomial_func, baseline_readings, baseline_weights)

    # Print these optimal values
    a, b, c = params
    print(f"Best fit: a={a}, b={b}, c={c}")

    # Testing the function with the given reading
    predicted_weight = polynomial_func(number, a, b, c)
    rounded_weight = max(0, round(predicted_weight / 2.5) * 2.5)
    return rounded_weight

if __name__ == "__main__":
    print(get_estimate(680))