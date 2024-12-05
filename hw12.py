import math
from scipy.stats import geom, binom, norm
from sympy import symbols, solve, diff, integrate


# Function 1: PMF of Y = X^2 for Discrete Uniform Distribution
def pmf_y_from_x_squared(values, y):
    pmf_x = 1 / len(values)  # Uniform PMF
    result = sum(pmf_x for x in values if x**2 == y)
    return result


# Function 2: PMF of Y = 7/sqrt(X) (Geometric Distribution)
def pmf_y_geometric(p, y):
    x = 7 / y
    if x.is_integer() and x > 0:
        return geom.pmf(int(x), p)
    return 0


# Function 3: CDF of Y = 5/sqrt(X) (Geometric Distribution)
def cdf_y_geometric(p, y):
    cdf_value = 0
    for yi in range(1, y + 1):
        x = 5 / yi
        if x.is_integer() and x > 0:
            cdf_value += geom.pmf(int(x), p)
    return cdf_value


# Function 4: PDF Transformation for Y = X^3
def pdf_transform_y_x3_1():
    # This is a multiple-choice question.
    return "f_Y(y) = (1/12) * y^(-2/3), -8 ≤ y ≤ 8"


# Function 5: PDF Transformation for Y = X^3 with f_X(x) = ωx^3
def pdf_transform_y_x3_2():
    # This is a multiple-choice question.
    return "f_Y(y) = (1/3) * y^(1/3), 0 ≤ y ≤ (2^(3/2))"


# Function 6: Jacobian Determinant
def jacobian_determinant():
    # Jacobian matrix determinant
    return 5 * 8 - (-5) * 8


# Function 7: Joint PDF Transformation
def joint_pdf(f_x1_x2, y1, y2):
    if y1 <= 0 or y2 <= 0:  # Check valid ranges
        return 0
    determinant = 1 / abs(y1)  # Jacobian determinant for transformation
    return f_x1_x2 * determinant


# Function 8: Expected Value of Binomial Random Variable
def expected_value_binomial(p, n, k):
    return k * n * p


# Function 9: Variance of Linear Combination of Binomial Variables
def variance_linear_comb(p, n, coeffs):
    var_single = binom.var(n, p)  # Variance of a single binomial variable
    variance = sum((c**2) * var_single for c in coeffs)
    return variance


# Function 10: Variance of Sample Mean
def variance_sample_mean(sigma, n):
    # Multiple-choice question; the answer is sigma^2 / n
    return "σ²/n"


# Function 11: Probability for Sample Variance
def probability_sample_variance(n, sigma, threshold):
    alpha = (n - 1) * sigma**2
    scale = sigma**2
    z = threshold / scale
    return norm.cdf(z)


# Function 12: Central Limit Theorem Approximation
def clt_probability(mean, std_dev, sample_size, bound):
    sample_std = std_dev / math.sqrt(sample_size)
    z = (bound - mean) / sample_std
    return norm.cdf(z)


# Main function to compute results
if __name__ == "__main__":
    # Input Values
    result_1 = pmf_y_from_x_squared(range(-9, 10), 16)
    result_2 = pmf_y_geometric(0.36, 14)
    result_3 = cdf_y_geometric(0.17, 10)
    result_4 = pdf_transform_y_x3_1()
    result_5 = pdf_transform_y_x3_2()
    result_6 = jacobian_determinant()
    result_7 = joint_pdf(1 / 2, -7, 4)
    result_8 = expected_value_binomial(0.52, 19, 8)
    result_9 = variance_linear_comb(0.32, 16, [1, 2, -3])
    result_10 = variance_sample_mean(3, 12)
    result_11 = probability_sample_variance(12, 3, 19.68)
    result_12 = clt_probability(1 / 4, math.sqrt(1 / 16), 9, 1 / 5)

    # Display results
    results = {
        "Result 1": result_1,
        "Result 2": result_2,
        "Result 3": result_3,
        "Result 4": result_4,
        "Result 5": result_5,
        "Result 6": result_6,
        "Result 7": result_7,
        "Result 8": result_8,
        "Result 9": result_9,
        "Result 10": result_10,
        "Result 11": result_11,
        "Result 12": result_12,
    }

    for key, value in results.items():
        print(f"{key}: {value}")
