import numpy as np
from stochasticSim import geometricBrownianMotion as gbm
from stochasticSim import plot_paths
import math

INTEREST_RATE = 0.057

# ---------------------------------------------------------------------------------------------------  

class gbmOption:
    def __init__(self, gbmObj, option_type, option_parameters):
        self.underlying = gbmObj
        self.option_type = option_type
        self.option_parameters = option_parameters

        self.option_parameters.validate(self.option_type)

        option_pricers = {
            "Vanilla European": self._vanilla_european_price,
            "Vanilla American": self._vanilla_american_price,
            "Vanilla Asian": self._vanilla_asian_price,
            "Barrier": self._barrier_price,
            # ...
        }

        self.price = option_pricers[self.option_type]()

    def _vanilla_european_price(self):
        #Analytical solution
        S = self.underlying.S0
        r = INTEREST_RATE
        sigma = self.underlying.sigma
        K = self.option_parameters.strike
        T = self.option_parameters.expiry

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        call_price = S * norm_cdf(d1) - K * math.exp(-r*T) * norm_cdf(d2)
        put_price = call_price + math.exp(-r*T)*K - S #put call parity
        return np.array([call_price, put_price])
    
    def _vanilla_american_price(self):
        #Least-Squares Monte Carlo to price American put (Longstaff Schwartz)
        r = INTEREST_RATE
        T = self.option_parameters.expiry
        dt = 1/504 #Twice a day
        N = 100000
        paths = self.underlying.simulate(T, dt, N)
        paths = paths.T # Do this nicer, be consistent
        K = self.option_parameters.strike
        steps = int(T / dt)
        
        # Discount factor per step
        discount_factor = np.exp(-r * dt)

        # Payoff matrix
        payoffs = np.maximum(K - paths, 0)
        
        # Cashflow matrix (initialize with final payoffs)
        cashflows = np.zeros_like(payoffs)
        cashflows[-1] = payoffs[-1]

        # Backward induction using least squares regression
        for t in range(steps - 1, 0, -1):
            in_the_money = payoffs[t] > 0
            X = paths[t, in_the_money]
            Y = cashflows[t + 1, in_the_money] * discount_factor
            
            if len(X) == 0:
                continue

            # Basis functions (here we use polynomial basis up to degree 2)
            A = np.vstack([X**i for i in range(3)]).T
            beta = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation_value = np.dot(A, beta)
            
            exercise = payoffs[t, in_the_money] > continuation_value
            cashflows[t, in_the_money] = np.where(exercise, payoffs[t, in_the_money], cashflows[t + 1, in_the_money] * discount_factor)

        # Price is the discounted average of the first row of cashflows
        put_price = np.mean(cashflows[1]) * discount_factor
        eu_price = self._vanilla_european_price() # Holds under the assumption that (r-q) > 0 ?? Not always true maybe
        return np.array([eu_price[0], put_price])
    
    def _vanilla_asian_price(self):
        #Monte Carlo
        r = INTEREST_RATE
        T = self.option_parameters.expiry
        dt = 1/504 #Twice a day
        N = 100000
        paths = self.underlying.simulate(T, dt, N)
        paths = paths.T # Do this nicer, be consistent
        K = self.option_parameters.strike
        
        # Calculate the average price for each path
        avg_prices = np.mean(paths, axis=0)
        call_payoffs = np.maximum(avg_prices - K, 0)
        put_payoffs = np.maximum(K - avg_prices, 0)

        # Discount the payoff back to present value
        discount_factor = np.exp(-r * T)
        call_price = discount_factor * np.mean(call_payoffs)
        put_price = discount_factor * np.mean(put_payoffs)
        
        return np.array([call_price, put_price])

    def _barrier_price(self):
        r = INTEREST_RATE
        T = self.option_parameters.expiry
        dt = 1/252
        N = 10000 #Monte carlo error roughly < 2e-3
        paths = self.underlying.simulate(T, dt, N)
        paths = paths.T # Do this nicer, be consistent
        K = self.option_parameters.strike
        barrier_type = self.option_parameters.extra_params['barrier_type']
        barrier_level = self.option_parameters.extra_params['barrier']

        # Check barrier conditions
        if barrier_type == 'up-and-in':
            hit_barrier = np.any(paths >= barrier_level, axis=0)
        elif barrier_type == 'down-and-in':
            hit_barrier = np.any(paths <= barrier_level, axis=0)
        elif barrier_type == 'up-and-out':
            hit_barrier = np.any(paths >= barrier_level, axis=0)
        elif barrier_type == 'down-and-out':
            hit_barrier = np.any(paths <= barrier_level, axis=0)
        else:
            raise ValueError("Invalid barrier type. Use 'up-and-in', 'down-and-in', 'up-and-out', or 'down-and-out'.")

        # Calculate the payoff for in-the-money paths
        call_payoffs = np.maximum(paths[-1] - K, 0)
        put_payoffs = np.maximum(K - paths[-1], 0)

        # Apply barrier conditions
        if 'in' in barrier_type:
            call_payoffs = call_payoffs * hit_barrier  # Only pay if barrier was hit
            put_payoffs = put_payoffs * hit_barrier  # Only pay if barrier was hit
        else:
            call_payoffs = call_payoffs * ~hit_barrier  # Only pay if barrier was not hit
            put_payoffs = put_payoffs * ~hit_barrier  # Only pay if barrier was not hit

        # Debug print statements
        print(f"Barrier type: {barrier_type}, Barrier level: {barrier_level}")
        print(f"Hit barrier: {np.mean(hit_barrier)}")  # Proportion of paths hitting the barrier

        # Discount the payoff back to present value
        discount_factor = np.exp(-r * T)
        call_price = discount_factor * np.mean(call_payoffs)
        put_price = discount_factor * np.mean(put_payoffs)

        return np.array([call_price, put_price])

# ---------------------------------------------------------------------------------------------------  

class optionParameters:
    def __init__(self, strike, expiry, **kwargs):
        self.strike = strike
        self.expiry = expiry
        self.extra_params = kwargs

    def validate(self, option_type):
        # Validate parameters are present and correct for option type
        if not (isinstance(self.strike, (int, float)) and self.strike > 0):
            raise ValueError("Invalid strike price")
        if not isinstance(self.expiry, (int, float)):
            raise ValueError("Invalid expiry date")
        
        validation_methods = {
            "Vanilla European": self._validate_vanilla_european,
            "Vanilla American": self._validate_vanilla_american,
            "Vanilla Asian": self._validate_vanilla_asian,
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
        
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2)

def test_calc():
    # Sample parameters
    S0 = 100  # Initial stock price
    mu = 0.057  # Drift
    sigma = 0.2  # Volatility
    strike = 110  # Strike price
    expiry = 1  # Time to expiration (1 year)

    # Create GBM and option parameters objects
    gbmObj = gbm(S0=S0, mu=mu, sigma=sigma)
    params_european =optionParameters(strike=strike, expiry=expiry)
    params_american = optionParameters(strike=strike, expiry=expiry)
    params_asian = optionParameters(strike=strike, expiry=expiry)
    params_barrier_in = optionParameters(strike=strike, expiry=expiry, barrier=180, barrier_type='up-and-in')
    params_barrier_out = optionParameters(strike=strike, expiry=expiry, barrier=180, barrier_type='up-and-out')

    # Create option objects
    european_option = gbmOption(gbmObj, "Vanilla European", params_european)
    american_option = gbmOption(gbmObj, "Vanilla American", params_american)
    asian_option = gbmOption(gbmObj, "Vanilla Asian", params_asian)
    barrier_option_in = gbmOption(gbmObj, "Barrier", params_barrier_in)
    barrier_option_out = gbmOption(gbmObj, "Barrier", params_barrier_out)

    print(f"European Call/Put Price: {european_option.price}")
    print(f"American Call/Put Price: {american_option.price}")
    print(f"Asian Call/Put Price: {asian_option.price}")
    print(f"Barrier Up-and-in Call/Put Price: {barrier_option_in.price}")
    print(f"Barrier Up-and-out Call/Put Price: {barrier_option_out.price}")
    print(european_option.price[0] - (barrier_option_in.price[0] + barrier_option_out.price[0]))
    # Perform checks
    assert american_option.price[0] >= european_option.price[0], "American call should be at least as expensive as European call."
    assert asian_option.price[0] <= european_option.price[0], "Asian call should be cheaper than European call."
    assert np.isclose(european_option.price[0], barrier_option_in.price[0] + barrier_option_out.price[0], rtol=0.01), "European call price does not equal the sum of up-and-out and up-and-in call prices."
    

# ---------------------------------------------------------------------------------------------------  

def main():
    test_calc()


if __name__ == "__main__":
    main()