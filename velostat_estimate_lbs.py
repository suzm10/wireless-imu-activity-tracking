from scipy.optimize import curve_fit
import numpy as np

def polynomial_func(reading, a, b, c):
    return a * reading ** 2 + b * reading + c

def get_estimate(number):
    baseline_weights = np.array([0, 2.5, 5, 10, 25])
    baseline_readings = np.array([400, 480, 520, 580, 675])

    params, covariance = curve_fit(polynomial_func, baseline_readings, baseline_weights)

    a, b, c = params
    print(f"Best fit: a={a}, b={b}, c={c}")

    if number < 350:
        print("Warning: Velostat reading is below the expected value with no weights! Expect strange results!")

    predicted_weight = polynomial_func(number, a, b, c)
    rounded_weight = max(0, round(predicted_weight / 2.5) * 2.5)
    return rounded_weight

if __name__ == "__main__":
    print(get_estimate(680))