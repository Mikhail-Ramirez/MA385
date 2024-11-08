import math
import numpy as np

# Function for Problem 1
def compute_joint_cdf(p_coin1, flips_coin1, p_coin2, flips_coin2, x, y):
    # Compute probabilities for each coin's success rate
    p_x = sum(math.comb(flips_coin1, k) * (p_coin1 ** k) * ((1 - p_coin1) ** (flips_coin1 - k)) for k in range(x + 1))
    p_y = sum(math.comb(flips_coin2, k) * (p_coin2 ** k) * ((1 - p_coin2) ** (flips_coin2 - k)) for k in range(y + 1))
    
    # Joint CDF value
    return round(p_x * p_y, 2)

# Function for Problem 2
def find_constant_c_for_cdf(a, b, x_max, y_max):
    # Calculate the normalization constant for the given cumulative distribution function
    total_sum = sum(math.exp((x ** 2) / a + (y ** 3) / b) for x in range(1, x_max + 1) for y in range(1, y_max + 1))
    c = 1 / total_sum
    return round(c, 2)

# Function for Problem 3
def find_constant_c_for_pmf(a, b, x_max, y_max):
    # Calculate the normalization constant for the given probability mass function
    total_sum = sum(((x / a) + (y / b)) for x in range(1, x_max + 1) for y in range(1, y_max + 1))
    c = 1 / total_sum
    return round(c, 2)

# Function for Problem 4
def compute_joint_pmf_heads(p_heads, heads_x, heads_y, x, y):
    # Use binomial distribution to find probabilities for each event
    p_x = math.comb(x + heads_x - 1, heads_x - 1) * (p_heads ** heads_x) * ((1 - p_heads) ** (x))
    p_y = math.comb(y + heads_y - 1, heads_y - 1) * (p_heads ** heads_y) * ((1 - p_heads) ** (y))
    
    # Joint PMF
    return round(p_x * p_y, 3)

# Function for Problem 5
def expectation_y(p_heads, die_sides, x_die_sides, y_heads):
    # E[Y] when X can be 1 (heads) or 0 (tails)
    e_y_given_heads = sum(i * (1 / die_sides) for i in range(1, die_sides + 1))
    e_y_given_tails = sum(i * (1 / x_die_sides) for i in range(1, x_die_sides + 1))
    e_y = p_heads * e_y_given_heads + (1 - p_heads) * e_y_given_tails
    return round(e_y, 2)

# Function for Problem 6
def expected_payout(num_00, num_01, num_10, num_11):
    total_balls = num_00 + num_01 + num_10 + num_11
    payout = (0 * num_00 + 1 * num_01 + 1 * num_10 + 2 * num_11) / total_balls
    return round(payout, 3)

# Function for Problem 7
def marginal_pmf_x(c, max_sum, x_value):
    # Calculate f_X(x) by summing over all y values for fixed x such that x + y <= max_sum
    sum_y_values = sum(c for y in range(1, max_sum - x_value + 1))
    return round(sum_y_values, 2)

# Function for Problem 8
def marginal_pmf_y(c, max_sum, y_value):
    # Calculate f_Y(y) by summing over all x values for fixed y such that x + y <= max_sum
    sum_x_values = sum(c * (x + y_value) for x in range(max_sum - y_value + 1))
    return round(sum_x_values, 2)


# Function for Problem 9: Determine Independence
def check_independence(pmf_function):
    """
    Determines if the given pmf function implies independence or dependence
    based on its structure.
    
    Parameters:
    - pmf_function (str): A description of the PMF function type.
    
    Returns:
    - str: "Independent" or "Dependent" based on the PMF structure.
    """
    # Independence test: Check for multiplicative form or conditional restrictions
    if "c" in pmf_function and any(op in pmf_function for op in ["*", "abs", " "]) and " + " not in pmf_function:
        return "Independent"
    return "Dependent"

# Function for Problem 10: Compute Correlation Coefficient
def correlation_coefficient(c, a_vals, b_vals, pmf_vals):
    """
    Calculates the correlation coefficient for two random variables X and Y
    given their PMF and possible values.
    
    Parameters:
    - c (float): Constant in the PMF function.
    - a_vals (list): Possible values for variable X.
    - b_vals (list): Possible values for variable Y.
    - pmf_vals (function): Lambda function representing the PMF formula.
    
    Returns:
    - float: Correlation coefficient rounded to two decimal places.
    """
    # Compute expectations
    x_expectation = sum(a * pmf_vals(c, a, b) for a in a_vals for b in b_vals)
    y_expectation = sum(b * pmf_vals(c, a, b) for a in a_vals for b in b_vals)
    
    # Compute variances
    x_variance = sum((a - x_expectation) ** 2 * pmf_vals(c, a, b) for a in a_vals for b in b_vals)
    y_variance = sum((b - y_expectation) ** 2 * pmf_vals(c, a, b) for a in a_vals for b in b_vals)
    
    # Compute covariance
    covariance = sum((a - x_expectation) * (b - y_expectation) * pmf_vals(c, a, b) for a in a_vals for b in b_vals)
    
    # Calculate correlation coefficient
    corr_coeff = covariance / (np.sqrt(x_variance) * np.sqrt(y_variance))
    return round(corr_coeff, 2)



# Main function to set up and call each problem function
def main():
    # Problem 1 constants and call
    p_coin1 = 0.3
    flips_coin1 = 5
    p_coin2 = 0.8
    flips_coin2 = 4
    result_1 = compute_joint_cdf(p_coin1, flips_coin1, p_coin2, flips_coin2, 2, 3)
    print(f"Problem 1 Result: {result_1}")

    # Problem 2 constants and call
    a = 19
    b = 25
    x_max, y_max = 5, 5
    result_2 = find_constant_c_for_cdf(a, b, x_max, y_max)
    print(f"Problem 2 Result: {result_2}")

    # Problem 3 constants and call
    a = 44
    b = 39
    x_max, y_max = 5, 5
    result_3 = find_constant_c_for_pmf(a, b, x_max, y_max)
    print(f"Problem 3 Result: {result_3}")

    # Problem 4 constants and call
    p_heads = 0.44
    heads_x = 4
    heads_y = 2
    result_4 = compute_joint_pmf_heads(p_heads, heads_x, heads_y, 6, 3)
    print(f"Problem 4 Result: {result_4}")
    
    # Problem 5 constants and call
    p_heads = 0.39
    die_sides = 6
    x_die_sides = 4
    result_5 = expectation_y(p_heads, die_sides, x_die_sides, 1)
    print(f"Problem 5 Result: {result_5}")

    # Problem 6 constants and call
    num_00, num_01, num_10, num_11 = 21, 20, 20, 19
    result_6 = expected_payout(num_00, num_01, num_10, num_11)
    print(f"Problem 6 Result: {result_6}")

    # Problem 7 constants and call
    c = 1 / 78  # Assuming c is determined by the normalization condition
    max_sum = 12
    x_value = 3
    result_7 = marginal_pmf_x(c, max_sum, x_value)
    print(f"Problem 7 Result: {result_7}")

    # Problem 8 constants and call
    c = 1 / (10 * (10 + 1) / 2)  # Given expression for c
    max_sum = 10
    y_value = 4
    result_8 = marginal_pmf_y(c, max_sum, y_value)
    print(f"Problem 8 Result: {result_8}")
    
    # Problem 9 constants and call
    pmf_functions = [
        "f(x, y) = c, where x + y <= 10",
        "f(x, y) = c3 * abs(x - y), where x >= 2, y >= 3",
        "f(x, y) = c, where x <= 10, y <= 10",
        "f(x, y) = cxy, where x <= 10, y <= 10",
        "f(x, y) = c(3x^3 + 5 * y), where x > 5, y > 5",
        "f(x, y) = c, where x <= 10, y <= 20"
    ]
    
    results_9 = [(pmf, check_independence(pmf)) for pmf in pmf_functions]
    print("Problem 9 Results:")
    for pmf, result in results_9:
        print(f"{pmf} -> {result}")

    # Problem 10 constants and call
    c = 1  # example constant, adjust based on normalization requirements
    a_vals = [-2, -1, 0, 1, 2]
    b_vals = [-1, 0, 1]
    pmf_vals = lambda c, a, b: c * (4 * abs(a) + 6 * b)
    
    result_10 = correlation_coefficient(c, a_vals, b_vals, pmf_vals)
    print(f"Problem 10 Result: {result_10}")

if __name__ == "__main__":
    main()

