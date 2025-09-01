import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

config = {
    "stocks": ["AAPL", "MSFT", "TSLA"], #List of stocks on the portafolio. 
    "initialWeights": np.array([0.1, 0.2]), #Weight of every stock on the portfolio. Numpy array
    "riskFreeRate": 0.03 #monthly risk-free rate, as expected returns are also monthly
}

def sharpeRatio (weights, expected_returns, cov_matrix, rf = 0):
    
    weights = weights / np.sum(weights)  #Normalize weights so they sum to 1

    portafolio_returns = expected_returns @ weights
    portafolio_variance = weights @ cov_matrix @ weights

    return (portafolio_returns - rf) / np.sqrt(portafolio_variance)

def sharpeRatioGradient (weights, expected_returns, cov_matrix, rf = 0, error = 1e-6):

    gradient = []

    for i in range(len(weights)):

        weights_plus = weights.copy() #Copy weights to create a new array and add the delta
        weights_minus = weights.copy() #Copy weights to create a new array and substract the delta

        weights_plus[i] += error
        weights_minus[i] -= error

        partial_derivative = (sharpeRatio(weights_plus, expected_returns, cov_matrix, rf) - sharpeRatio(weights_minus, expected_returns, cov_matrix, rf)) / (2 * error)

        gradient.append(partial_derivative) 

    return np.array(gradient)

def maxSharpeRatio (weights, expected_returns, cov_matrix, rf = 0, learning_rate = 0.0001, max_iterations = 100000, tolerance = 1e-9, allow_short = False):

    sharpe_history = []
    weights_history = []

    weights = weights / np.sum(weights) #Normalize weights so they sum to 1

    for i in range(max_iterations): 
        current_sharpe = sharpeRatio(weights, expected_returns, cov_matrix, rf)
        gradient = sharpeRatioGradient(weights, expected_returns, cov_matrix, rf)

        sharpe_history.append(current_sharpe)
        weights_history.append(weights.copy())

        ####### No-short restriction #######
        if not allow_short:
            weights[weights < 0] = 0 #If a weight is negative, set it to 0

            mask = (weights == 0) & (gradient < 0) #If a weight is 0 and its gradient is negative, set the gradient to 0 so it doesn't get updated
            gradient[mask] = 0

        new_weights = weights + learning_rate * gradient
        new_weights = new_weights / np.sum(new_weights) #Normalize weights so they sum to 1

        sharpe_change = abs(sharpeRatio(new_weights, expected_returns, cov_matrix, rf) - current_sharpe)

        if sharpe_change < tolerance: break

        weights = new_weights

    final_sharpe = sharpeRatio(weights, expected_returns, cov_matrix, rf)

    return {
        "optimal_weights": weights,
        "optimal_sharpe": final_sharpe,
        "iterations": i,
        "sharpe_history": sharpe_history,
        "weights_history": weights_history
    }

def plotSharpeOptimization(init_weights, expected_returns, cov_matrix, rf = 0, learning_rate = 0.0001, max_iterations = 10000000, tolerance = 1e-9, allow_short = False, resolution = 100):

    if len(init_weights) != 3:
        raise ValueError("Initial weights must be a 3-element array.")

    # Normalize initial weights to sum to 1
    init_weights_norm = init_weights / np.sum(init_weights)
    
    # Run optimization algorithm
    results = maxSharpeRatio(init_weights_norm, expected_returns, cov_matrix, rf, learning_rate=learning_rate, max_iterations=max_iterations, tolerance=tolerance, allow_short=allow_short)

    # Extract optimization trajectory
    trajectory = np.array(results["weights_history"])
    w1_path, w2_path = trajectory[:, 0], trajectory[:, 1]
    
    # Calculate dynamic grid bounds based on trajectory with 20% margin
    w1_min, w1_max = w1_path.min(), w1_path.max()
    w2_min, w2_max = w2_path.min(), w2_path.max()
    
    # Expand bounds to include default range and add margin
    w1_bounds = (min(w1_min, -2), max(w1_max, 2))
    w2_bounds = (min(w2_min, -2), max(w2_max, 2))
    
    w1_margin = (w1_bounds[1] - w1_bounds[0]) * 0.2
    w2_margin = (w2_bounds[1] - w2_bounds[0]) * 0.2
    
    # Create high-resolution mesh grid
    w1_range = np.linspace(w1_bounds[0] - w1_margin, w1_bounds[1] + w1_margin, resolution)
    w2_range = np.linspace(w2_bounds[0] - w2_margin, w2_bounds[1] + w2_margin, resolution)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    
    # Compute Sharpe ratio surface
    sharpe_surface = np.full_like(W1, np.nan)
    max_sharpe_clip = 2.5  # Clip extreme values for better visualization
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w1, w2 = W1[i, j], W2[i, j]
            w3 = 1 - w1 - w2
            weights_grid = np.array([w1, w2, w3])
            
            try:
                sharpe_val = sharpeRatio(weights_grid, expected_returns, cov_matrix, rf)
                # Clip extreme values for better visualization
                if np.abs(sharpe_val) <= max_sharpe_clip:
                    sharpe_surface[i, j] = sharpe_val
            except (ZeroDivisionError, ValueError, np.linalg.LinAlgError):
                # Handle numerical issues gracefully
                continue
    
    # Calculate Sharpe values for trajectory points
    sharpe_path = [sharpeRatio(w, expected_returns, cov_matrix, rf) for w in trajectory]
    
    # Create and configure 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot with improved settings
    surf = ax.plot_surface(
        W1, W2, sharpe_surface, 
        cmap='viridis', alpha=0.75, 
        antialiased=True, edgecolor='none',
        linewidth=0, rasterized=True  # Better performance for high resolution
    )
    
    # Add optimization trajectory with enhanced styling
    ax.plot(w1_path, w2_path, sharpe_path, 
           color='red', linewidth=3, alpha=0.9, 
           marker='.', markersize=4, label='Optimization Path')
    
    # Add enhanced trajectory markers
    # Starting point
    ax.scatter(w1_path[0], w2_path[0], sharpe_path[0], 
              color='lime', s=150, alpha=1.0, 
              label=f'Start (S: {sharpe_path[0]:.3f})', 
              edgecolors='darkgreen', linewidths=2)
    
    # Optimal point
    ax.scatter(w1_path[-1], w2_path[-1], sharpe_path[-1], 
              color='red', s=150, alpha=1.0, 
              label=f'Optimum (S: {sharpe_path[-1]:.3f})', 
              edgecolors='darkred', linewidths=2)
    
    # Configure plot appearance
    ax.set_xlabel('Weight₁ (w₁)', fontsize=12, labelpad=10)
    ax.set_ylabel('Weight₂ (w₂)', fontsize=12, labelpad=10)
    ax.set_zlabel('Sharpe Ratio', fontsize=12, labelpad=10)
    ax.set_title('Sharpe Ratio Surface with Optimization Trajectory', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add improved colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Sharpe Ratio Value', fontsize=11)
    
    # Set optimal viewing angle
    ax.view_init(elev=65, azim=-50)
    
    # Add legend with better positioning
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Improve grid appearance
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


if len(config["initialWeights"]) == len(config["stocks"])-1 :
   config["initialWeights"] = np.append(config["initialWeights"], 1 - np.sum(config["initialWeights"]))

precios = yf.download(config["stocks"], period="5y", interval="1mo") #Download 5 years of monthly data for the stocks on the config
returns = precios['Close'].pct_change()[1:] #Monthly returns 

expected_returns = returns.mean().to_numpy() #Expected monthly return based on 5y history
cov_matrix = returns.cov().to_numpy()




# gradient = sharpeRatioGradient(config["initialWeights"],
#                                expected_returns,
#                                cov_matrix,
#                                config["riskFreeRate"],
#                                error=1e-9)

# max_sharpe = maxSharpeRatio(config["initialWeights"],
#                            expected_returns,
#                            cov_matrix,
#                            config["riskFreeRate"],
#                            learning_rate=0.0001,
#                            max_iterations=1000000,
#                            tolerance=1e-9,
#                            allow_short=False)

sharpe_ratio = sharpeRatio(config["initialWeights"],
                          expected_returns,
                          cov_matrix,
                          config["riskFreeRate"])

plot_results = plotSharpeOptimization(config["initialWeights"],
                       expected_returns,
                       cov_matrix, 
                       config["riskFreeRate"],
                       learning_rate=0.0001,
                       max_iterations=10000000,
                       tolerance=1e-9,
                       allow_short=True,
                       resolution=100)

print("Initial Weights: ", config["initialWeights"])
print(f"Initial Sharpe Ratio: {sharpe_ratio:.6f}")
print(f"Optimal Weights: {plot_results['optimal_weights']}")
print(f"Optimal Sharpe Ratio: {plot_results['optimal_sharpe']:.6f}")
print(f"Weights Sum Verification: {np.sum(plot_results['optimal_weights']):.8f}")