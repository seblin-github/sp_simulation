import numpy as np
import matplotlib.pyplot as plt
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

        C = S * norm_cdf(d1) - K * math.exp(-r*T) * norm_cdf(d2)
        P = C + math.exp(-r*T)*K - S
        return np.array([C, P])

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
            # ...
        }
        
        # Call the appropriate validation method
        if option_type in validation_methods:
            validation_methods[option_type]()
            print("Success!")
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def _validate_vanilla_european(self):
        #Additional validation can go here
        pass

    def _validate_vanilla_american(self):       
        # Specific validation for American options
        if 'early_exercise' not in self.extra_params:
            raise ValueError("American options require an 'early_exercise' parameter")
        
def norm_cdf(x):
    # Compute CDF for standard normal distribution
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_pdf(x):
    # Compute PDF for standard normal distribution
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2)

def main():
    S0 = 100
    mu = 0.08
    sigma = 0.02

    gbmObj = gbm(S0, mu, sigma)
    option_type = "Vanilla European"
    K = 102
    T = 1
    vanilla_parameters = optionParameters(K, T)
    myOption = gbmOption(gbmObj, option_type, vanilla_parameters)
    plot_paths(gbmObj.simulate(T, 1/365, 10))
    print(myOption.price)


if __name__ == "__main__":
    main()