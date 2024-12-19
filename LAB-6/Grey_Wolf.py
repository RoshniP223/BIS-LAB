import numpy as np
# Initialize parameters
n_wolves = 5
max_iter = 100
bounds = [-10, 10]
a = 2
X_min, X_max = bounds
# Initialize wolves' positions randomly
wolf_positions = np.random.uniform(X_min, X_max, (n_wolves, 1))

# Fitness function (f(x) = x^2 + 2x + 1)
def fitness_function(x):
    return x**2 + 2*x + 1

# Update leaders (alpha, beta, delta)
def update_leaders(fitness, positions):
    sorted_idx = np.argsort(fitness)
    return positions[sorted_idx[0]], positions[sorted_idx[1]], positions[sorted_idx[2]]

# Main loop
for t in range(max_iter):
    a_t = 2 - t * (2 / max_iter)  # Linearly decreasing a
    
    # Evaluate fitness of wolves
    fitness = np.array([fitness_function(wolf) for wolf in wolf_positions])

    # Update leaders
    alpha, beta, delta = update_leaders(fitness, wolf_positions)

    # Move wolves towards the leaders
    for i in range(n_wolves):
        A_alpha, C_alpha = 2 * a_t * np.random.random() - a_t, 2 * np.random.random()
        A_beta, C_beta = 2 * a_t * np.random.random() - a_t, 2 * np.random.random()
        A_delta, C_delta = 2 * a_t * np.random.random() - a_t, 2 * np.random.random()

        # Calculate new position
        D_alpha = abs(C_alpha * alpha - wolf_positions[i])
        D_beta = abs(C_beta * beta - wolf_positions[i])
        D_delta = abs(C_delta * delta - wolf_positions[i])
        wolf_positions[i] = np.clip((alpha - A_alpha * D_alpha + beta - A_beta * D_beta + delta - A_delta * D_delta) / 3, X_min, X_max)

# Output the best solution found
alpha_fitness = fitness[np.argmin(fitness)]
print("Optimal solution:", alpha)
print("Fitness of optimal solution:", alpha_fitness)
