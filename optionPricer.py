import numpy as np
from stochasticSim import geometricBrownianMotion as gbm
from stochasticSim import plot_paths
import math

INTEREST_RATE = 0.057

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
            # ...
        }

        self.price = option_pricers[self.option_type]()

    def _vanilla_european_price(self):
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
        K = self.option_parameters.strike
        T = self.option_parameters.expiry
        dt = 1/504 #Twice a day
        N = 100000
        paths = self.underlying.simulate(T, dt, N)
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
        r = INTEREST_RATE
        T = self.option_parameters.expiry
        dt = 1/504 #Twice a day
        N = 100000
        paths = self.underlying.simulate(T, dt, N)
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
            # ...
        }
        
        # Call the appropriate validation method
        if option_type in validation_methods:
            validation_methods[option_type]()
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def _validate_vanilla_european(self):
        #Additional validation can go here
        pass

    def _validate_vanilla_american(self):       
        #Additional validation can go here
        pass
    def _validate_vanilla_asian(self):       
        #Additional validation can go here
        pass
        
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2)

def test_calc():
    # Sample parameters
    S0 = 100  # Initial stock price
    mu = 0.05  # Drift
    sigma = 0.2  # Volatility
    strike = 110  # Strike price
    expiry = 1  # Time to expiration (1 year)

    # Create GBM and option parameters objects
    gbmObj = gbm(S0=S0, mu=mu, sigma=sigma)
    params_european =optionParameters(strike=strike, expiry=expiry)
    params_american = optionParameters(strike=strike, expiry=expiry)
    params_asian = optionParameters(strike=strike, expiry=expiry)

    # Create option objects
    european_option = gbmOption(gbmObj, "Vanilla European", params_european)
    american_option = gbmOption(gbmObj, "Vanilla American", params_american)
    asian_option = gbmOption(gbmObj, "Vanilla Asian", params_asian)

    print(f"European Call Price: {european_option.price}")
    print(f"American Call Price: {american_option.price}")
    print(f"Asian Call Price: {asian_option.price}")

    # Perform checks
    assert american_option.price[0] >= european_option.price[0], "American call should be at least as expensive as European call."
    assert asian_option.price[0] <= european_option.price[0], "Asian call should be cheaper than European call."

def main():
    test_calc()


if __name__ == "__main__":
    main()