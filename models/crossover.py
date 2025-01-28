import copy 

def crossover(parent1, parent2, rng_seed, crossover_rate=0.5):
    """
    Perform crossover between two individuals.
    This works by swapping some parameters of the two fuzzy neural networks.
    The parameters we consider are parent.neuron_weights
    crossover_rate: float, probability of swapping 
    """
    child = copy_individual(parent1)
    local_parent_2 = copy_individual(parent2) #itherwise it shares the same memory address from here on

    # Swap neuron weights
    if child.update_gene == "neuron_weights":
        for i in range(len(child.neuron_weights)):
            if rng_seed.random() < crossover_rate:
                child.neuron_weights[i] = local_parent_2.neuron_weights[i]

    # Swap last layer weights
    if child.update_gene == "V":
        if rng_seed.random() < crossover_rate:
            child.V = local_parent_2.V
    return child


def copy_individual(individual):
    """
    Create a deep copy of an individual.
    """
    new_individual = copy.deepcopy(individual)
    new_individual.fitness = None
    return new_individual
