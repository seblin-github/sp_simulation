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

class heston:
    def __init__(self, S0, V0, mu, kappa, theta, sigma, rho):
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
        
        Z1 = np.random.normal(0, 1, (N, num_steps))
        Z2 = np.random.normal(0, 1, (N, num_steps))
        W1 = Z1
        W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        # Simulate the paths
        for t in range(1, num_steps + 1):
            V[:, t] = np.maximum(V[:, t-1] + self.kappa * (self.theta - V[:, t-1]) * dt + 
                                    self.sigma * np.sqrt(V[:, t-1]) * np.sqrt(dt) * W2[:, t-1], 0)
            S[:, t] = S[:, t-1] * np.exp((self.mu - 0.5 * V[:, t-1]) * dt + 
                                            np.sqrt(V[:, t-1]) * np.sqrt(dt) * W1[:, t-1])
        
        # Save the paths to the class attribute
        self.paths = S
        self.vPaths = V
        
        return self.paths

class ornsteinUhlenbeck:
    def __init__(self,S0, theta, mu, sigma, jump_intensity=None, jump_mean=None, jump_std=None):
        # Optional jump process (for electricity price simulation) 
        self.S = S0
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.paths = None
        
    def simulate(self, T, dt, N, jump=True):
        timesteps = int(T / dt)
        self.paths = np.zeros((N, timesteps+1))
        
        for i in range(N):
            X = np.zeros(timesteps+1)
            X[0] = self.S
            
            for j in range(1, timesteps+1):
                dW = np.sqrt(dt) * np.random.randn()
                X[j] = X[j-1] + self.theta * (self.mu - X[j-1]) * dt + self.sigma * dW
                
                if jump and self.jump_intensity is not None and self.jump_mean is not None and self.jump_std is not None:
                    if np.random.rand() < self.jump_intensity * dt:
                        jump_size = np.random.normal(self.jump_mean, self.jump_std)
                        X[j] += jump_size
            
            self.paths[i, :] = X
        
        return np.linspace(0., T, timesteps+1), self.paths

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
    return -1

def main():
    S0 = 100
    mu = 0.02
    sigma = 0.9
    theta = 2
    jump_i = 0.5
    jump_mean = 1
    jump_std = 0.1
    ornsteinObj = ornsteinUhlenbeck(S0, theta, S0*(1 + mu), sigma, jump_i, jump_mean, jump_std)

    T = 2
    dt = 1/365
    N = 100

    ornsteinObj.simulate(T, dt, N)
    plot_paths(ornsteinObj.paths)

if __name__ == "__main__":
    main()
