# Importing the Python Libraries necessary for the assignment.
import numpy as np # Used to generate numbers for the population.
import matplotlib.pyplot as plt # Used for Plot Visualisation
import csv #To log the results in a csv file
import os  #To create the folder to store the images
os.makedirs("plots", exist_ok=True)

# Define the Rastrigin function
def rastrigin(X):
    x, y = X
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    
# Note: Population Size across each algorithm is 50 and it runs for 100 generations on all of them.

# Implementation of the Genetic Algorithm (GA)
class GeneticAlgorithm:
    def __init__(self, func, bounds, pop_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1):
      # Setting the parameters
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
      # Randomly initializing the population within the bounds
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
      # Tracks the best fitness based on history from generation
        self.best_fitness_history = []

      # Randomly picks 3 individuals from the population and returns teh best with the lowest fitness
    def select_parents(self):
        selected = np.random.choice(range(self.pop_size), 3, replace=False)
        best = min(selected, key=lambda idx: self.func(self.population[idx]))
        return self.population[best]

      # Crossover between the parents to produce a child and ensures the child stays within bounds  
    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = 0.5
            child = parent1 + alpha * (parent2 - parent1) * (np.random.rand(2) - 0.5)
        else:
            child = parent1
        return np.clip(child, self.bounds[0], self.bounds[1])

      # Applies mutation based on the mutation rate and ensures the mutated individual is within bounds
    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation_strength = 0.1 * (self.bounds[1] - self.bounds[0])
            individual += np.random.uniform(-mutation_strength, mutation_strength, 2)
        return np.clip(individual, self.bounds[0], self.bounds[1])

      # Tracks best fitness in every generation
    def run(self):
        for _ in range(self.generations):
            new_population = []
            for _ in range(self.pop_size):
                p1 = self.select_parents()
                p2 = self.select_parents()
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = np.array(new_population)
            best = min(self.population, key=self.func)
            self.best_fitness_history.append(self.func(best))
        return self.population[np.argmin([self.func(ind) for ind in self.population])]

# Implementation of the Differential Evolution (DE)
class DifferentialEvolution:
    def __init__(self, func, bounds, pop_size=50, generations=100, F=0.5, CR=0.9):
      # Setting the parameters
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.F = F
        self.CR = CR
        # Initialize population uniformly within the bounds
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
        # To track best fitness value found at each generation
        self.best_fitness_history = []

    def run(self):
        # Returns the best solution found after all generations
        for _ in range(self.generations):
            new_population = []
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                # Randomly select three distinct individuals a, b, c from the population
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                # Mutation: create mutant vector by adding scaled difference between b and c to a within the bounds
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                # Crossover: for each dimension, choose mutant or target gene based on crossover probability CR
                trial = np.where(np.random.rand(2) < self.CR, mutant, self.population[i])
                # Selection: if trial vector has better (lower) fitness, it replaces the current individual
                if self.func(trial) < self.func(self.population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(self.population[i])
                # Updates the population for the next generation
            self.population = np.array(new_population)
                # Record best fitness value found in current population
            best = min(self.population, key=self.func)
                # Returns the best solution after all generations have been run
            self.best_fitness_history.append(self.func(best))
        return self.population[np.argmin([self.func(ind) for ind in self.population])]

# Implementation of the Particle Swarm Optimization (PSO)
class ParticleSwarmOptimization:
      # Setting the parameters
    def __init__(self, func, bounds, pop_size=50, generations=100, w=0.7, c1=1.5, c2=1.5):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.w = w         # inertia weight
        self.c1 = c1       # cognitive (self) coefficient
        self.c2 = c2       # social (swarm) coefficient

        # Initialize particle positions uniformly within the bounds
        self.positions = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
        # Initialize particle velocities to zero vectors
        self.velocities = np.zeros((pop_size, 2))
        # Initialize personal best positions as the initial positions
        self.pbest = self.positions.copy()
        # Evaluate fitness for each particle's personal best position
        self.pbest_val = np.array([self.func(p) for p in self.positions])
        # Determine the global best position among all personal bests
        self.gbest = self.pbest[np.argmin(self.pbest_val)]
        # Keep track of best fitness value found at each generation
        self.best_fitness_history = []

    def run(self):
        for _ in range(self.generations):
            for i in range(self.pop_size):
              # Generate two random factors for stochastic influence
                r1, r2 = np.random.rand(), np.random.rand()
              # Calculate cognitive velocity component: attraction to personal best
                cognitive = self.c1 * r1 * (self.pbest[i] - self.positions[i])
              # Calculate social velocity component: attraction to global best
                social = self.c2 * r2 * (self.gbest - self.positions[i])
              # Update velocity combining inertia, cognitive and social components
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
              # Update position by adding velocity and keep it within bounds
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], self.bounds[0], self.bounds[1])
              # Calculate fitness at new position
                fit = self.func(self.positions[i])
              # Update personal best if current fitness is better
                if fit < self.pbest_val[i]:
                    self.pbest[i] = self.positions[i]
                    self.pbest_val[i] = fit
              # Identify the global best position among all personal bests
            best_idx = np.argmin(self.pbest_val)
            self.gbest = self.pbest[best_idx]
              # Record the global best fitness for this generation
            self.best_fitness_history.append(self.func(self.gbest))
              # Return the best position found after all iterations
        return self.gbest

# Run All Algorithms
# Setting the boundary over the domain
bounds = (-5.12, 5.12)
generations = 100

ga = GeneticAlgorithm(rastrigin, bounds, generations=generations)
ga_solution = ga.run()

de = DifferentialEvolution(rastrigin, bounds, generations=generations)
de_solution = de.run()

pso = ParticleSwarmOptimization(rastrigin, bounds, generations=generations)
pso_solution = pso.run()

# Print Final Solutions and Fitness for Each
print(f"GA Solution: {ga_solution}, Fitness: {rastrigin(ga_solution)}")
print(f"DE Solution: {de_solution}, Fitness: {rastrigin(de_solution)}")
print(f"PSO Solution: {pso_solution}, Fitness: {rastrigin(pso_solution)}")

# Individual Convergence Plots

# GA Convergence Plot
plt.figure(figsize=(8, 5))
plt.plot(ga.best_fitness_history, color='blue')
plt.title('Genetic Algorithm Convergence')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/ga_convergence.png")
plt.show()

# DE Convergence Plot
plt.figure(figsize=(8, 5))
plt.plot(de.best_fitness_history, color='green')
plt.title('Differential Evolution Convergence')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/de_convergence.png")
plt.show()

# PSO Convergence Plot
plt.figure(figsize=(8, 5))
plt.plot(pso.best_fitness_history, color='orange')
plt.title('Particle Swarm Optimization Convergence')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/pso_convergence.png")
plt.show()

# Plot Combined Convergence of all Algorithms
plt.figure(figsize=(10, 6))
plt.plot(ga.best_fitness_history, label='Genetic Algorithm')
plt.plot(de.best_fitness_history, label='Differential Evolution')
plt.plot(pso.best_fitness_history, label='Particle Swarm Optimization')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Convergence of GA, DE, and PSO on Rastrigin Function')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/combined_convergence.png")
plt.show()

# Save results to CSV
with open("results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Algorithm", "Best X", "Best Y", "Best Fitness"])
    writer.writerow(["Genetic Algorithm", ga_solution[0], ga_solution[1], rastrigin(ga_solution)])
    writer.writerow(["Differential Evolution", de_solution[0], de_solution[1], rastrigin(de_solution)])
    writer.writerow(["Particle Swarm Optimization", pso_solution[0], pso_solution[1], rastrigin(pso_solution)])
      
# Saves fitness per generation for all algorithms to a CSV file
with open("fitness_log.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write header row
    writer.writerow(["Generation", "GA Fitness", "DE Fitness", "PSO Fitness"])
    
    # Iterate over all generations and write fitness values
    for gen in range(generations):
        ga_fit = ga.best_fitness_history[gen] if gen < len(ga.best_fitness_history) else ""
        de_fit = de.best_fitness_history[gen] if gen < len(de.best_fitness_history) else ""
        pso_fit = pso.best_fitness_history[gen] if gen < len(pso.best_fitness_history) else ""
        writer.writerow([gen + 1, ga_fit, de_fit, pso_fit])
    
