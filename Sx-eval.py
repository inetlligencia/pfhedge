import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

# Parameters
S0 = 100       # Initial stock price
K = 100        # Strike price
sigma = 0.2    # Volatility
r = 0.05       # Risk-free rate
T = 1          # Time to maturity (in years)

# Bounds for the large move
S_low = S0
S_high = S0 * (1 + 3 * sigma)

# Function to compute d1 given stock price S'
def d1(S_prime):
    return (np.log(S_prime / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

# Function to compute gamma given stock price S'
def gamma(S_prime):
    d1_val = d1(S_prime)
    return norm.pdf(d1_val) / (S_prime * sigma * np.sqrt(T))

# Log-normal probability density function of the stock price
def lognormal_pdf(S_prime):
    exponent = -((np.log(S_prime / S0) - (r - 0.5 * sigma ** 2) * T) ** 2) / (2 * sigma ** 2 * T)
    return (1 / (S_prime * sigma * np.sqrt(2 * np.pi * T))) * np.exp(exponent)

# Integrand for the expected delta change
def integrand(S_prime):
    return gamma(S_prime) * (S_prime - S0) * lognormal_pdf(S_prime)

# Numerical integration
expected_realized_delta, error = quad(integrand, S_low, S_high)

expected_realized_delta
