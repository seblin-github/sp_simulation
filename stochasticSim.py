import numpy as np
import matplotlib.pyplot as plt

class geometricBrownianMotion:
    def __init__(self, S0, mu, sigma):
        self.S0 = S0        #Spot price at t=0
        self.mu = mu        #Drift of gBM
        self.sigma = sigma  #volatility of gBM
        self.paths = None   #Initialize paths for simulation
    
    def simulate(self, T, dt, N):
        # T (float): end of simulation t=T
        # dt (float): stepsize (yrs)
        # N (int): simulations

        num_steps = int(T/dt)
        paths = np.zeros((N, num_steps + 1))
        paths[:, 0] = self.S0

        for t in range(1, num_steps + 1):
            W = np.random.normal(0, 1, N)
            paths[:, t] = paths[:, t -1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * W)

        self.paths = paths
        return paths
    
    def plot_paths(self, num_paths=10):
        # plot simulates paths of gBM
        # num_paths (int): Number of paths to plot.

        if self.paths is None:
            raise ValueError("No paths to plot. Run simulate() first.")

        T = self.paths.shape[1] - 1
        dt = T / (self.paths.shape[1] - 1)
        
        plt.figure(figsize=(10, 6))
        for i in range(min(num_paths, self.paths.shape[0])):
            plt.plot(np.linspace(0, T * dt, T + 1), self.paths[i])
        plt.title('Geometric Brownian Motion Simulation')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()


def main():
    S0 = 100
    mu = 0.07
    sigma = 0.4
    gbm = geometricBrownianMotion(S0, mu, sigma)

    T = 2
    dt = 1/365
    N = 100
    gbm.simulate(T, dt, N)
    gbm.plot_paths()
    

if __name__ == "__main__":
    main()
