import math
from scipy.integrate import dblquad

# New approach for conditional variance for question 1 with different logic
def conditional_variance_x_given_y_new_method(c, y_val):
    # Define range of x values satisfying 0 <= x + y <= 16 with Y=10
    valid_x_values = [x for x in range(17) if 0 <= x + y_val <= 16]
    n = len(valid_x_values)

    # Compute new normalizing constant by dividing c by the number of valid values
    c_val = c / n if n != 0 else 0  # Avoid division by zero if no valid values

    # Compute E(X|Y) and E(X^2|Y) using the new distribution approach
    sum_x = sum([x * c_val for x in valid_x_values])
    sum_x_sq = sum([(x ** 2) * c_val for x in valid_x_values])
    
    # Compute variance using the standard formula
    variance = sum_x_sq - (sum_x ** 2)
    return round(variance, 2)

# New approach for conditional variance for question 2 with adjusted parameters
def conditional_variance_y_given_x_new_method(c, x_val):
    # Define range of y values satisfying 0 <= x + y <= 20 with X=16
    valid_y_values = [y for y in range(21) if 0 <= x_val + y <= 20]
    n = len(valid_y_values)

    # Compute new normalizing constant by dividing c by the number of valid values
    c_val = c / n if n != 0 else 0  # Avoid division by zero if no valid values

    # Compute E(Y|X) and E(Y^2|X) using the new distribution approach
    sum_y = sum([y * c_val for y in valid_y_values])
    sum_y_sq = sum([(y ** 2) * c_val for y in valid_y_values])
    
    # Compute variance using the standard formula
    variance = sum_y_sq - (sum_y ** 2)
    return round(variance, 2)

# Adjusted probability region calculation for question 3
def probability_region_new_method(c):
    # Compute P(2 <= X <= 6, 1 <= Y) by integrating the given continuous PDF
    func = lambda x, y: c * math.exp(-x - y)
    prob, _ = dblquad(func, 2, 6, lambda y: 1, lambda y: math.inf)
    return round(prob, 3)

# New approach for probability for question 4 with changed boundaries
def probability_x_less_than_17y_new_method(c):
    # Define probability limits for P(X <= 17Y)
    func = lambda x, y: c * math.exp(-x - y)
    prob, _ = dblquad(func, 0, math.inf, lambda x: 0, lambda x: x / 17)
    return round(prob, 3)

# Adjusted expectation calculation for question 5
def expectation_2x_new_method(c):
    # Define expectation E(2X) for f(x, y) = 6e^(-3x - 2y)
    func = lambda x, y: 2 * x * c * math.exp(-3*x - 2*y)
    expectation, _ = dblquad(func, 0, math.inf, lambda y: 0, lambda y: math.inf)
    return round(expectation, 2)

# Main function to compute and print the results with the new approaches
def main_new_method():
    # Define constants based on the updated problem statements
    c1, y_val = 1, 10          # Problem 1 constants
    c2, x_val = 1, 16          # Problem 2 constants
    c3 = 1                     # Problem 3 and 4 constants
    c4 = 6                     # Problem 5 constant

    # Call each function with the new methods
    var_x_given_y_new = conditional_variance_x_given_y_new_method(c1, y_val)
    var_y_given_x_new = conditional_variance_y_given_x_new_method(c2, x_val)
    prob_region_new = probability_region_new_method(c3)
    prob_x_less_than_17y_new = probability_x_less_than_17y_new_method(c3)
    expected_2x_new = expectation_2x_new_method(c4)

    # Print the new results
    print(f"New Variance of X given Y=10: {var_x_given_y_new}")
    print(f"New Variance of Y given X=16: {var_y_given_x_new}")
    print(f"New Probability P(2 <= X <= 6, 1 <= Y): {prob_region_new}")
    print(f"New Probability P(X <= 17Y): {prob_x_less_than_17y_new}")
    print(f"New Expectation E(2X): {expected_2x_new}")

# Execute the new method main function
main_new_method()

