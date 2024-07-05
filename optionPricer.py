import numpy as np
from stochasticSim import geometricBrownianMotion as gbm
from stochasticSim import heston
import scipy as scp
import math

INTEREST_RATE = 0.057

# ---------------------------------------------------------------------------------------------------  
# - - - Option class, collects model, option type, option parameters and price
class stockOption:
    def __init__(self, gbmObj, option_type, option_parameters):
        self.underlying = gbmObj
        self.option_type = option_type
        self.option_parameters = option_parameters

        self.option_parameters.validate(self.option_type)

        if gbmObj.identifier == "gbm":
            option_pricers = {
                "European": gbm_european_asol,
                "American": gbm_american_fd,
                "Asian": asian_mc,
                "Barrier": barrier_mc,
                # ...
            }
        elif gbmObj.identifier == "heston":
            option_pricers = {
                "European": european_mc,
                "American": american_mc,
                "Asian": asian_mc,
                "Barrier": barrier_mc,
                # ...
            }
        else:
            raise ValueError(f"Unknown underlying model: {option_type}")

        self.price = option_pricers[self.option_type](self)

# ---------------------------------------------------------------------------------------------------  
# - - - Option parameters class, includes validate function
class optionParameters:
    def __init__(self, strike, expiry, option_direction, **kwargs):
        self.strike = strike
        self.expiry = expiry
        self.option_direction = option_direction
        self.extra_params = kwargs

    def validate(self, option_type):
        # Validate parameters are present and correct for option type
        if not (isinstance(self.strike, (int, float)) and self.strike > 0):
            raise ValueError("Invalid strike price")
        if not isinstance(self.expiry, (int, float)):
            raise ValueError("Invalid expiry date")
        if not isinstance(self.option_direction, (str)):
            raise ValueError("Invalid option direction")
        
        validation_methods = {
            "European": self._validate_vanilla_european,
            "American": self._validate_vanilla_american,
            "Asian": self._validate_vanilla_asian,
            "Barrier": self._validate_barrier,
            # ...
        }
        
        # Call the appropriate validation method
        if option_type in validation_methods:
            validation_methods[option_type]()
        else:
            raise ValueError(f"Unknown option type: {option_type}")
        
    def __str__(self):
        return f"OptionParameters(strike={self.strike}, expiry={self.expiry}, extra_params={self.extra_params})"
    
    def __repr__(self):
        return self.__str__()

    def _validate_vanilla_european(self):
        #Additional validation can go here
        pass

    def _validate_vanilla_american(self):       
        #Additional validation can go here
        pass
    def _validate_vanilla_asian(self):       
        #Additional validation can go here
        pass
    def _validate_barrier(self):
        if 'barrier' not in self.extra_params or not isinstance(self.extra_params['barrier'], (int, float)):
            raise ValueError("Barrier options require a 'barrier' parameter")
        if 'barrier_type' not in self.extra_params or not isinstance(self.extra_params['barrier_type'], (str)):
            raise ValueError("Barrier options require a 'barrier_type' parameter")
        pass

# ---------------------------------------------------------------------------------------------------  
# - - - Analytical solution
def gbm_european_asol(optionObj):
    #Analytical solution
    S = optionObj.underlying.S0
    r = INTEREST_RATE
    sigma = optionObj.underlying.sigma
    K = optionObj.option_parameters.strike
    T = optionObj.option_parameters.expiry

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if optionObj.option_parameters.option_direction == "call":
        price = S * norm_cdf(d1) - K * math.exp(-r*T) * norm_cdf(d2)
    elif optionObj.option_parameters.option_direction == "put":
        price =  K * math.exp(-r*T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return price

# - - - Monte Carlo
def european_mc(optionObj):
    r = INTEREST_RATE
    T = optionObj.option_parameters.expiry
    dt = 1/252
    N = 10000 #Monte carlo error roughly < 2e-3
    strike = optionObj.option_parameters.strike
    paths = optionObj.underlying.simulate(T, dt, N)
    
    discount_factor = np.exp(-r * T)
    
    if optionObj.option_parameters.option_direction == 'call':
        payoff = np.maximum(paths[:, -1] - strike, 0)
    elif optionObj.option_parameters.option_direction == 'put':
        payoff = np.maximum(strike - paths[:, -1], 0)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    
    price = discount_factor * np.mean(payoff)
    
    return price
    
def american_mc(optionObj):
    # Longstaff - Schwartz
    # Parameters
    r = INTEREST_RATE
    T = optionObj.option_parameters.expiry
    dt = 1/252
    N = 10000  # Number of paths
    paths = optionObj.underlying.simulate(T, dt, N)
    K = optionObj.option_parameters.strike
    steps = int(T / dt)
    
    # Discount factor per step
    discount_factor = np.exp(-r * dt)

    # Payoff matrix
    if optionObj.option_parameters.option_direction == "call":
        payoffs = np.maximum(paths - K, 0)
    elif optionObj.option_parameters.option_direction == "put":
        payoffs = np.maximum(K - paths, 0)
    
    # Cashflow matrix (initialize with final payoffs)
    cashflows = np.zeros_like(payoffs)
    cashflows[:, -1] = payoffs[:, -1]
    # Backward induction using least squares regression
    for t in range(steps - 1, -1, -1):
        in_the_money = payoffs[:, t] > 0
        X = payoffs[in_the_money, t]
        Y = cashflows[in_the_money, t + 1] * discount_factor
        if len(X) == 0:
            continue

        # Basis functions (here we use polynomial basis up to degree 2)
        A = np.vstack([X**i for i in range(3)]).T
        beta = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation_value = np.dot(A, beta)
        
        exercise = payoffs[in_the_money, t] > continuation_value
        cashflows[in_the_money, t] = np.where(exercise, payoffs[in_the_money, t], cashflows[in_the_money, t + 1] * discount_factor)
        discount_path = cashflows[:, t] == 0
        cashflows[discount_path, t] = cashflows[discount_path, t + 1] * discount_factor

    # Price is the discounted average of the first row of cashflows
    price = np.mean(cashflows[:, 1]) * discount_factor
    return price

def asian_mc(optionObj):
    #Monte Carlo
    r = INTEREST_RATE
    T = optionObj.option_parameters.expiry
    dt = 1/252
    N = 10000
    paths = optionObj.underlying.simulate(T, dt, N)
    paths = paths.T # Do this nicer, be consistent
    K = optionObj.option_parameters.strike
    
    # Calculate the average price for each path
    avg_prices = np.mean(paths, axis=0)
    if optionObj.option_parameters.option_direction == "call":
        payoffs = np.maximum(avg_prices - K, 0)
    elif optionObj.option_parameters.option_direction == "put":
        payoffs = np.maximum(K - avg_prices, 0)

    # Discount the payoff back to present value
    discount_factor = np.exp(-r * T)
    price = discount_factor * np.mean(payoffs)
    return price

def barrier_mc(optionObj):
    r = INTEREST_RATE
    T = optionObj.option_parameters.expiry
    dt = 1/252
    N = 10000 #Monte carlo error roughly < 2e-3
    paths = optionObj.underlying.simulate(T, dt, N)
    K = optionObj.option_parameters.strike
    barrier_type = optionObj.option_parameters.extra_params['barrier_type']
    barrier_level = optionObj.option_parameters.extra_params['barrier']

    # Check barrier conditions
    if barrier_type == 'up-and-in':
        hit_barrier = np.any(paths >= barrier_level, axis=1)
    elif barrier_type == 'down-and-in':
        hit_barrier = np.any(paths <= barrier_level, axis=1)
    elif barrier_type == 'up-and-out':
        hit_barrier = np.any(paths >= barrier_level, axis=1)
    elif barrier_type == 'down-and-out':
        hit_barrier = np.any(paths <= barrier_level, axis=1)
    else:
        raise ValueError("Invalid barrier type. Use 'up-and-in', 'down-and-in', 'up-and-out', or 'down-and-out'.")

    # Calculate the payoff for in-the-money paths
    if optionObj.option_parameters.option_direction == "call":
        payoffs = np.maximum(paths[:, -1] - K, 0)
    elif optionObj.option_parameters.option_direction == "put":
        payoffs = np.maximum(K - paths[:, -1], 0)

    # Apply barrier conditions
    if 'in' in barrier_type:
        payoffs = payoffs * hit_barrier  # Only pay if barrier was hit
    else:
        payoffs = payoffs * ~hit_barrier  # Only pay if barrier was not hit

    # Discount the payoff back to present value
    discount_factor = np.exp(-r * T)
    price = discount_factor * np.mean(payoffs)
    return price

# - - - Finite difference
def gbm_american_fd(optionObj):
    S0 = optionObj.underlying.S0
    r = INTEREST_RATE
    sigma = optionObj.underlying.sigma
    K = optionObj.option_parameters.strike
    T = optionObj.option_parameters.expiry
    option_direction = optionObj.option_parameters.option_direction
    Smax = 6 * S0
    
    # Grid settings
    dt = 1 / 252
    M = int(T / dt)
    N = 200
    S = np.linspace(0, Smax, N+1)
    V = np.zeros((M+1, N+1))
    
    # Boundary conditions
    if option_direction == 'call':
        V[:, -1] = np.maximum(S[-1] - K, 0)  # Max stock price boundary
        V[:, 0] = 0  # Zero stock price boundary
        V[-1, :] = np.maximum(S - K, 0)  # Payoff at maturity
    elif option_direction == 'put':
        V[:, -1] = 0  # Max stock price boundary
        V[:, 0] = np.maximum(K - S[0], 0)  # Zero stock price boundary
        V[-1, :] = np.maximum(K - S, 0)  # Payoff at maturity
    else:
        raise ValueError("Option direction must be 'call' or 'put'")
    
    # Coefficients
    alpha = 0.25 * dt * (sigma**2 * (np.arange(N+1)**2) - r * np.arange(N+1))
    beta = -0.5 * dt * (sigma**2 * (np.arange(N+1)**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (np.arange(N+1)**2) + r * np.arange(N+1))
    
    # Tridiagonal matrix setup
    A = np.zeros((N-1, N-1))
    B = np.zeros((N-1, N-1))
    
    for i in range(1, N):
        if i > 1:
            A[i-1, i-2] = -alpha[i]
            B[i-1, i-2] = alpha[i]
        A[i-1, i-1] = 1 - beta[i]
        B[i-1, i-1] = 1 + beta[i]
        if i < N-1:
            A[i-1, i] = -gamma[i]
            B[i-1, i] = gamma[i]
    
    # Time stepping
    for j in range(M, 0, -1):
        b = B @ V[j, 1:N]
        V[j-1, 1:N] = np.linalg.solve(A, b)
        
        # Early exercise condition
        if option_direction == 'call':
            V[j-1, 1:N] = np.maximum(V[j-1, 1:N], S[1:N] - K)
        elif option_direction == 'put':
            V[j-1, 1:N] = np.maximum(V[j-1, 1:N], K - S[1:N])
    
    # Interpolation to get the option price at S0
    price = np.interp(S0, S, V[0, :])
    return price

def heston_american_fd(optionObj):
    # NOT WORKING !!!
    return -1
    S0 = optionObj.underlying.S0
    r = INTEREST_RATE
    sigma = optionObj.underlying.sigma
    kappa = optionObj.underlying.kappa
    theta = optionObj.underlying.theta
    rho = optionObj.underlying.rho
    v0 = optionObj.underlying.V0
    K = optionObj.option_parameters.strike
    T = optionObj.option_parameters.expiry
    option_type = optionObj.option_parameters.option_direction
    Smax = 6 * S0
    Vmax = 6 * v0
    
    # Grid settings
    dt = 1 / 252
    M = int(T / dt)
    N = 200
    P = 200
    dv = Vmax / P
    S = np.linspace(0, Smax, N+1)
    v = np.linspace(0, Vmax, P+1)
    V = np.zeros((M+1, N+1, P+1))
    
    # Initial conditions
    if option_type == 'call':
        V[-1, :, :] = np.maximum(S[:, None] - K, 0)  # Payoff at maturity
    elif option_type == 'put':
        V[-1, :, :] = np.maximum(K - S[:, None], 0)  # Payoff at maturity
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    
    # Precompute coefficients for the finite difference scheme
    a = np.zeros((N+1, P+1))
    b = np.zeros((N+1, P+1))
    c = np.zeros((N+1, P+1))
    d = np.zeros((N+1, P+1))
    e = np.zeros((N+1, P+1))
    f = np.zeros((N+1, P+1))

    for i in range(1, N):
        for j in range(1, P):
            a[i, j] = 0.25 * dt * (sigma**2 * v[j] * (i**2) - r * i)
            b[i, j] = -0.5 * dt * (sigma**2 * v[j] * (i**2) + r)
            c[i, j] = 0.25 * dt * (sigma**2 * v[j] * (i**2) + r * i)
            d[i, j] = 0.25 * dt * kappa * (theta - v[j]) - 0.25 * dt * rho * sigma * i * v[j]**0.5
            e[i, j] = -0.5 * dt * kappa * (theta - v[j]) + 0.5 * dt * rho * sigma * i * v[j]**0.5
            f[i, j] = 0.25 * dt * kappa * (theta - v[j]) + 0.25 * dt * rho * sigma * i * v[j]**0.5

    # Time stepping
    for m in range(M, 0, -1):
        for j in range(1, P):
            # Set up the tridiagonal matrix system for each variance step
            A = np.zeros((N-1, N-1))
            B = np.zeros((N-1, N-1))
            for i in range(1, N):
                if i > 1:
                    A[i-1, i-2] = -a[i, j]
                    B[i-1, i-2] = a[i, j]
                A[i-1, i-1] = 1 - b[i, j]
                B[i-1, i-1] = 1 + b[i, j]
                if i < N-1:
                    A[i-1, i] = -c[i, j]
                    B[i-1, i] = c[i, j]
            
            # Solve the linear system A * V_new = B * V_old for each variance step
            V_old = V[m, 1:N, j]
            V_new = np.linalg.solve(A, B @ V_old)

            # Apply early exercise condition
            if option_type == 'call':
                V_new = np.maximum(V_new, S[1:N] - K)
            elif option_type == 'put':
                V_new = np.maximum(V_new, K - S[1:N])
            
            V[m-1, 1:N, j] = V_new

        # Boundary conditions for S
        V[m-1, 0, :] = 0 if option_type == 'call' else K
        V[m-1, -1, :] = (Smax - K) if option_type == 'call' else 0

        # Boundary conditions for v
        V[m-1, :, 0] = 0 if option_type == 'call' else K
        V[m-1, :, -1] = (S - K) if option_type == 'call' else 0

    # Interpolation to get the option price at S0 and v0
    price = np.interp(S0, S, V[0, :, int(v0 / dv)])
    return price
# - - - Fourier transform

# ---------------------------------------------------------------------------------------------------  
# - - - Misc. functions     
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2)

def test_calc():
    # Sample parameters
    S0 = 100  # Initial stock price
    mu = 0.057  # Drift
    sSigma = 0.2  # Volatility
    theta = sSigma ** 2 # Long-term variance
    kappa = 1.57 # Rate of mean reversion
    rho = -0.7 # Correlation between Wiener of stock and variance
    vSigma = 0.3 # Volatility of volatility

    strike = 100 # Strike price
    expiry = 1  # Time to expiration (1 year)

    # Create GBM and option parameters objects
    gbmObj = gbm(S0=S0, mu=mu, sigma=sSigma)
    params_european =optionParameters(strike=strike, expiry=expiry, option_direction = "call")
    params_bin =optionParameters(strike=strike, expiry=expiry, option_direction = "call", barrier_type = "up-and-in", barrier = 100)
    params_bout =optionParameters(strike=strike, expiry=expiry, option_direction = "call", barrier_type = "up-and-out", barrier = 100)

    # Create option objects
    european_option = stockOption(gbmObj, "European", params_european)
    bin_option = stockOption(gbmObj, "Barrier", params_bin)
    bout_option = stockOption(gbmObj, "Barrier", params_bout)

    print(european_option.price)
    print(bin_option.price + bout_option.price)

# ---------------------------------------------------------------------------------------------------  

def main():
    test_calc()


if __name__ == "__main__":
    main()