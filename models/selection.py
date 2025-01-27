import random
from models.crossover import *
from typing import List # Add at top of file
from models.models import FNNModel


def selection(
        population: List[FNNModel],
        selection_mu,
        selection_lambda,
        mutation_rate,
        crossover_rate,
        rng_seed,
        fitness_type,
        x,
        y,
        data_encoding,
        pred_method,
        map_class_dict,
        selection_strategy="comma",
        method="tournament",
        tournament_size=3
        ):
    """
    Select the best individuals from the population, eventually with mutation.
    population: list of Individual
    selection_mu: int, number of individuals to select from the population
    selection_lambda: int, number of individuals to generate
    mutation_rate: float, probability of mutation
    selection_strategy: str, either "plus" or "comma"
    x, y: input-output data for fitness calculation
    """
    
    if selection_mu > selection_lambda:
        raise ValueError("selection_lambda must be greater than selection_mu")
    if selection_strategy not in ["plus", "comma"]:
        raise ValueError("selection_strategy must be either 'plus' or 'comma'")
    
    # Calculate fitness of population
    for individual in population:
        if individual.fitness is None:
            individual.calculate_fitness(fitness_type, x, y, data_encoding, pred_method, map_class_dict,fast=True)[fitness_type]

    offspring:List[FNNModel] = generate_offspring(population, selection_lambda, mutation_rate, crossover_rate, rng_seed, method=method, tournament_size=tournament_size)
    
    # Calculate fitness of offspring
    for individual in offspring:
        individual.generate_parameters(x, y)
        individual.calculate_fitness(fitness_type, x, y, data_encoding, pred_method, map_class_dict, fast=True)[fitness_type]

    # Select the best individuals according to the selection strategy
    if selection_strategy == "plus":
        offspring += population
    offspring.sort(key=lambda x: x.fitness, reverse=True)
    
    return offspring[:selection_mu]


def tournament(population, rng_seed, tournament_size):
    """
    Select the best individual from a random subset of the population.
    population: list of individuals
    tournament_size: int, number of individuals in the subset
    """
    subset = rng_seed.choice(population, tournament_size, replace=False)
    return max(subset, key=lambda x: x.fitness)


def generate_offspring(parents, selection_lambda, mutation_rate, crossover_rate, rng_seed, method="tournament", tournament_size=3):
    """
    Generate new individuals from the parents, with probability of parent selection proportional to fitness.
    parents: list of Individual
    selection_lambda: int, number of individuals to generate
    mutation_rate: float, probability of mutation
    method: str, either "tournament" or "proportional"
    """
    if method not in ["tournament", "proportional"]:
        raise ValueError("method must be either 'tournament' or 'proportional'")
    
    if method == "proportional":
        # Compute the probability of selecting each parent
        fitness_sum = sum(p.fitness for p in parents)
        parent_probabilities = [p.fitness / fitness_sum for p in parents]

    offspring = []
    for _ in range(selection_lambda):
        if method == "tournament":
            parent1 = tournament(parents, rng_seed, tournament_size=tournament_size)
            parent2 = tournament(parents, rng_seed, tournament_size=tournament_size)
        else:
            parent1 = rng_seed.choices(parents, weights=parent_probabilities)[0]
            parent2 = rng_seed.choices(parents, weights=parent_probabilities)[0]
        child = crossover(parent1, parent2, rng_seed, crossover_rate )
        child.mutate(mutation_rate=mutation_rate)
        offspring.append(child)
    return offspring
