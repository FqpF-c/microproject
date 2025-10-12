import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Callable, Optional
import logging
from sklearn.metrics import accuracy_score, f1_score
import copy
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EquilibriumOptimizer:
    """
    Equilibrium Optimization (EO) algorithm for hyperparameter tuning.
    Bio-inspired optimization algorithm based on equilibrium states in physics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_iterations = config.get('max_iterations', 30)
        self.population_size = config.get('population_size', 50)
        self.dimensions = config.get('dimensions', 4)
        self.bounds = config.get('bounds', {})

        # EO specific parameters
        self.a1 = 2.0  # Acceleration factor
        self.a2 = 1.0  # Acceleration factor
        self.GP = 0.5  # Generation probability

        # Population and fitness tracking
        self.population = None
        self.fitness_values = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []

        # Equilibrium pool
        self.equilibrium_pool = None

        logger.info(f"Initialized EO with {self.population_size} particles for {self.max_iterations} iterations")

    def _initialize_population(self) -> np.ndarray:
        """Initialize random population within bounds."""
        population = np.zeros((self.population_size, self.dimensions))

        # Map bounds to parameters
        param_names = ['batch_size', 'learning_rate', 'hidden_units', 'dropout_rate']

        for i in range(self.population_size):
            for j, param_name in enumerate(param_names):
                if param_name in self.bounds:
                    low, high = self.bounds[param_name]
                    if param_name == 'batch_size' or param_name == 'hidden_units':
                        # Integer parameters
                        population[i, j] = np.random.randint(low, high + 1)
                    else:
                        # Float parameters
                        population[i, j] = np.random.uniform(low, high)

        return population

    def _decode_solution(self, solution: np.ndarray) -> Dict[str, Any]:
        """Decode numerical solution to hyperparameters."""
        param_names = ['batch_size', 'learning_rate', 'hidden_units', 'dropout_rate']
        hyperparams = {}

        for i, param_name in enumerate(param_names):
            if param_name == 'batch_size':
                hyperparams[param_name] = int(solution[i])
            elif param_name == 'hidden_units':
                # For simplicity, use the value as the main hidden unit size
                # Real implementation might have more complex encoding
                hyperparams[param_name] = [int(solution[i]), int(solution[i] // 2), int(solution[i] // 4)]
            else:
                hyperparams[param_name] = float(solution[i])

        return hyperparams

    def _ensure_bounds(self, solution: np.ndarray) -> np.ndarray:
        """Ensure solution stays within bounds."""
        param_names = ['batch_size', 'learning_rate', 'hidden_units', 'dropout_rate']

        for i, param_name in enumerate(param_names):
            if param_name in self.bounds:
                low, high = self.bounds[param_name]
                solution[i] = np.clip(solution[i], low, high)

        return solution

    def _update_equilibrium_pool(self, iteration: int):
        """Update equilibrium pool with best solutions."""
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_values)

        # Select top 4 solutions
        n_best = min(4, len(sorted_indices))
        best_indices = sorted_indices[:n_best]

        # Create equilibrium pool
        self.equilibrium_pool = self.population[best_indices].copy()

        # Add average of best solutions
        avg_solution = np.mean(self.equilibrium_pool, axis=0)
        self.equilibrium_pool = np.vstack([self.equilibrium_pool, avg_solution])

    def _exponential_term(self, t: float, iteration: int) -> float:
        """Calculate exponential term for EO update."""
        t0 = np.random.random()
        return -1 * (self.a1 - 1) * np.random.random() * t0 / (np.exp(self.a2 * t) - 1)

    def _concentration_update(self, solution: np.ndarray, eq_candidate: np.ndarray,
                            iteration: int) -> np.ndarray:
        """Update concentration using EO formula."""
        t = (1 - iteration / self.max_iterations) ** (self.a2 * iteration / self.max_iterations)

        r = np.random.random()
        if r < self.GP:
            # Random concentration update
            GCP = 0.5 * r * np.ones(self.dimensions) * (r >= self.GP)
            G0 = GCP * (eq_candidate - solution * r)
            G = G0 * self._exponential_term(t, iteration)
        else:
            # Normal concentration update
            F = self.a1 * np.sign(r - 0.5) * (np.exp(-1 * r * t) - 1)
            G = GCP * (eq_candidate - solution * r) if r < 0.5 else F

        # Update solution
        new_solution = eq_candidate + (solution - eq_candidate) * F + \
                      G / r / 100 * np.random.random(self.dimensions)

        return new_solution

    def _evaluate_fitness(self, hyperparams: Dict[str, Any],
                         fitness_function: Callable) -> float:
        """Evaluate fitness of hyperparameters."""
        try:
            fitness = fitness_function(hyperparams)
            return fitness
        except Exception as e:
            logger.error(f"Error evaluating fitness: {e}")
            return float('inf')  # Return worst possible fitness

    def optimize(self, fitness_function: Callable,
                verbose: bool = True) -> Tuple[Dict[str, Any], float]:
        """
        Run the Equilibrium Optimization algorithm.

        Args:
            fitness_function: Function that takes hyperparameters and returns fitness score
            verbose: Whether to print progress

        Returns:
            Tuple of (best_hyperparams, best_fitness)
        """
        logger.info("Starting Equilibrium Optimization...")

        # Initialize population
        self.population = self._initialize_population()
        self.fitness_values = np.full(self.population_size, float('inf'))

        # Evaluate initial population
        for i in range(self.population_size):
            hyperparams = self._decode_solution(self.population[i])
            fitness = self._evaluate_fitness(hyperparams, fitness_function)
            self.fitness_values[i] = fitness

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.population[i].copy()

        if verbose:
            logger.info(f"Initial best fitness: {self.best_fitness:.4f}")

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Update equilibrium pool
            self._update_equilibrium_pool(iteration)

            # Update each particle
            for i in range(self.population_size):
                # Select random equilibrium candidate
                eq_index = np.random.randint(0, len(self.equilibrium_pool))
                eq_candidate = self.equilibrium_pool[eq_index]

                # Update concentration
                new_solution = self._concentration_update(
                    self.population[i], eq_candidate, iteration
                )

                # Ensure bounds
                new_solution = self._ensure_bounds(new_solution)

                # Evaluate new solution
                new_hyperparams = self._decode_solution(new_solution)
                new_fitness = self._evaluate_fitness(new_hyperparams, fitness_function)

                # Update if better
                if new_fitness < self.fitness_values[i]:
                    self.population[i] = new_solution
                    self.fitness_values[i] = new_fitness

                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_solution.copy()

            # Store convergence
            self.convergence_curve.append(self.best_fitness)

            if verbose and iteration % 5 == 0:
                logger.info(f"Iteration {iteration}: Best fitness = {self.best_fitness:.4f}")

        # Return best solution
        best_hyperparams = self._decode_solution(self.best_solution)

        logger.info(f"Optimization completed. Best fitness: {self.best_fitness:.4f}")
        logger.info(f"Best hyperparameters: {best_hyperparams}")

        return best_hyperparams, self.best_fitness

    def get_convergence_curve(self) -> List[float]:
        """Get the convergence curve."""
        return self.convergence_curve


class HyperparameterTuner:
    """
    Hyperparameter tuner using Equilibrium Optimization.
    """

    def __init__(self, model_class, config: Dict[str, Any]):
        self.model_class = model_class
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training configuration
        self.max_epochs = config.get('max_epochs', 20)  # Reduced for tuning
        self.patience = config.get('patience', 5)

        # Initialize optimizer
        self.eo_optimizer = EquilibriumOptimizer(config.get('optimization', {}))

    def _create_model(self, hyperparams: Dict[str, Any]) -> nn.Module:
        """Create model with given hyperparameters."""
        model_config = self.config.copy()
        model_config.update(hyperparams)

        model = self.model_class(model_config)
        return model.to(self.device)

    def _train_and_evaluate(self, hyperparams: Dict[str, Any],
                           X_train: torch.Tensor, y_train: torch.Tensor,
                           X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Train model with hyperparameters and return validation fitness."""
        try:
            # Create model
            model = self._create_model(hyperparams)

            # Setup optimizer
            optimizer = optim.Adam(
                model.parameters(),
                lr=hyperparams.get('learning_rate', 0.001),
                weight_decay=hyperparams.get('l2_regularization', 0.001)
            )

            # Setup loss function
            criterion = nn.CrossEntropyLoss()

            # Training data
            batch_size = hyperparams.get('batch_size', 32)
            n_samples = len(X_train)
            n_batches = (n_samples + batch_size - 1) // batch_size

            best_val_loss = float('inf')
            patience_counter = 0

            # Training loop
            for epoch in range(self.max_epochs):
                model.train()
                total_loss = 0

                # Mini-batch training
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_samples)

                    batch_X = X_train[start_idx:end_idx]
                    batch_y = y_train[start_idx:end_idx]

                    optimizer.zero_grad()

                    if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                        # Model returns multiple outputs (like CA-LSTM)
                        outputs, _ = model(batch_X)
                    else:
                        outputs = model(batch_X)

                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                        val_outputs, _ = model(X_val)
                    else:
                        val_outputs = model(X_val)

                    val_loss = criterion(val_outputs, y_val).item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    break

            # Calculate final validation metrics
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                    val_outputs, _ = model(X_val)
                else:
                    val_outputs = model(X_val)

                val_preds = torch.argmax(val_outputs, dim=1)
                val_accuracy = accuracy_score(y_val.cpu().numpy(), val_preds.cpu().numpy())
                val_f1 = f1_score(y_val.cpu().numpy(), val_preds.cpu().numpy(), average='weighted')

            # Return negative F1 score (since we want to minimize)
            return -val_f1

        except Exception as e:
            logger.error(f"Error in training: {e}")
            return float('inf')  # Return worst fitness

    def tune(self, X_train: torch.Tensor, y_train: torch.Tensor,
             X_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, Any]:
        """
        Tune hyperparameters using Equilibrium Optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Best hyperparameters
        """
        logger.info("Starting hyperparameter tuning with Equilibrium Optimization...")

        # Convert to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        # Define fitness function
        def fitness_function(hyperparams: Dict[str, Any]) -> float:
            return self._train_and_evaluate(hyperparams, X_train, y_train, X_val, y_val)

        # Run optimization
        best_hyperparams, best_fitness = self.eo_optimizer.optimize(
            fitness_function=fitness_function,
            verbose=True
        )

        logger.info(f"Hyperparameter tuning completed!")
        logger.info(f"Best validation F1 score: {-best_fitness:.4f}")
        logger.info(f"Best hyperparameters: {best_hyperparams}")

        return best_hyperparams


if __name__ == "__main__":
    # Example usage
    from src.models.ca_lstm import ChannelAttentionLSTM

    config = {
        'input_dim': 150,
        'num_classes': 3,
        'optimization': {
            'max_iterations': 10,  # Reduced for testing
            'population_size': 20,
            'dimensions': 4,
            'bounds': {
                'batch_size': [16, 64],
                'learning_rate': [1e-4, 1e-2],
                'hidden_units': [32, 128],
                'dropout_rate': [0.1, 0.5]
            }
        }
    }

    # Create dummy data
    X_train = torch.randn(1000, 150)
    y_train = torch.randint(0, 3, (1000,))
    X_val = torch.randn(200, 150)
    y_val = torch.randint(0, 3, (200,))

    # Create tuner
    tuner = HyperparameterTuner(ChannelAttentionLSTM, config)

    # Run tuning
    # best_params = tuner.tune(X_train, y_train, X_val, y_val)