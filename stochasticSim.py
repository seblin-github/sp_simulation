import pandas as pd
import math

class geometricBrownianMotion:
    def __init__(self, S, mu, sigma):
        self.S = S          #Spot price at t=0
        self.mu = mu        #Drift of gBM
        self.sigma = sigma  #volatility of gBM
    
    def simulate(self, T, dt, N):
        # T: end of simulation t=T
        # dt: stepsize (yrs)
        # N: simulations
        return N


def main():
    print("Hello!")

if __name__ == "__main__":
    main()
