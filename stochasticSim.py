import numpy as np
import matplotlib.pyplot as plt
from sobolNormal import generate_standard_normal_matrix
import scipy.stats as ss

# .simulate returns matrix of paths (N by M)
# N: simulation
# M: timestep

class StochasticModel:
    def __init__(self, identifier):
        self.identifier = identifier

class geometricBrownianMotion(StochasticModel):
    def __init__(self, S0, mu, sigma):
        super().__init__("gbm")
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
        W_mat = np.random.randn(num_steps+1, N) #generate_standard_normal_matrix(num_steps+1,N)

        for t in range(1, num_steps + 1):
            W = W_mat[t, :]
            paths[:, t] = paths[:, t -1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * W)

        self.paths = paths
        return paths

class heston(StochasticModel):
    def __init__(self, S0, V0, mu, kappa, theta, sigma, rho):
        super().__init__("heston")
        self.S0 = S0            # Spot price at t=0
        self.V0 = V0            # Volatility at t=0
        self.mu = mu            # Drift of Heston
        self.kappa = kappa      # Rate of mean reversion
        self.theta = theta      # Long-term variance
        self.sigma = sigma      # Volatility of volatility
        self.rho = rho          # Correlation between Wiener of stock and variance
        self.paths = None      # Initialize paths for S simulation
        self.vPaths = None      # Initialize paths for V simulation

    def simulate(self, T, dt, N):
        # T (float): end of simulation t=T
        # dt (float): stepsize (yrs)
        # N (int): simulations

        num_steps = int(T / dt)
        S = np.zeros((N, num_steps + 1))
        V = np.zeros((N, num_steps + 1))
        S[:, 0] = self.S0
        V[:, 0] = self.V0
        
        Z1 = np.random.randn(N, num_steps+1) #Sobol needs to be reviewed for heston
        Z2 = np.random.randn(N, num_steps+1) #Sobol needs to be reviewed for heston
        W1 = Z1
        W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2   

        # Simulate the paths, Euler-Maruyama
        for t in range(1, num_steps + 1):
            V[:, t] = np.maximum(V[:, t-1] + self.kappa * (self.theta - V[:, t-1]) * dt + 
                                    self.sigma * np.sqrt(V[:, t-1]) * np.sqrt(dt) * W2[:, t-1], 0)
            S[:, t] = S[:, t-1] * np.exp((self.mu - 0.5 * V[:, t-1]) * dt + 
                                            np.sqrt(V[:, t-1]) * np.sqrt(dt) * W1[:, t-1])
        
        # Save the paths to the class attribute
        self.paths = S
        self.vPaths = V
        
        return self.paths

class ornsteinUhlenbeck(StochasticModel):
    def __init__(self,S0, theta, mu, sigma, jump_intensity=None, jump_mean=None, jump_std=None):
        # Optional jump process (for electricity price simulation) 
        super().__init__("Ornstein-Uhlenbeck")
        self.S = S0
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.paths = None
        
    def simulate(self, T, dt, N, jump=True):
        num_steps = int(T / dt)
        self.paths = np.zeros((N, num_steps+1))
        W_mat = generate_standard_normal_matrix(num_steps+1,N)
        
        # Simulate the paths, Euler-Maruyama
        for i in range(N):
            X = np.zeros(num_steps+1)
            X[0] = self.S
            
            for j in range(1, num_steps+1):
                dW = np.sqrt(dt) * W_mat[j, i]
                X[j] = X[j-1] + self.theta * (self.mu - X[j-1]) * dt + self.sigma * dW
                
                if jump and self.jump_intensity is not None and self.jump_mean is not None and self.jump_std is not None:
                    if np.random.rand() < self.jump_intensity * dt:
                        jump_size = np.random.normal(self.jump_mean, self.jump_std)
                        X[j] += jump_size
            
            self.paths[i, :] = X
        
        return self.paths

def plot_paths(paths, num_paths=10):
    # plot simulates paths of gBM
    # num_paths (int): Number of paths to plot.

    if paths is None:
        raise ValueError("No paths to plot. Run simulate() first.")

    T = paths.shape[1] - 1
    dt = T / (paths.shape[1] - 1)
    
    plt.figure(figsize=(10, 6))
    for i in range(min(num_paths, paths.shape[0])):
        plt.plot(np.linspace(0, T * dt, T + 1), paths[i])
    plt.title('Price Simulation')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

def main():
    S0 = 100
    mu = 102
    sigma = 0.9
    theta = 1
    jump_i = 0.5
    jump_mean = 0.1
    jump_std = 0.1
    ornsteinObj = ornsteinUhlenbeck(S0, theta, mu, sigma, jump_i, jump_mean, jump_std)

    S0 = 100  # Initial stock price
    mu = 0.05  # Drift
    sigma = 0.2  # Volatility
    theta = 0.3 # Long-term variance
    kappa = 0.1 # Rate of mean reversion
    rho = -0.7 # Correlation between Wiener of stock and variance
    vSigma = 0.3 # Volatility of volatility

    hestonObj = heston(S0= S0, V0=sigma, mu=mu, kappa=kappa, theta=theta,sigma=vSigma, rho= rho)

    T = 2
    dt = 1/365
    N = 100

    hestonObj.simulate(T, dt, N)
    plot_paths(hestonObj.paths)

if __name__ == "__main__":
    main()
