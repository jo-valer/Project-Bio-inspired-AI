import copy 

SWAP4ALL_MFS = True # If True, swap all gaussian parameters of a feature, for all its membership functions.


def crossover(parent1, parent2, rng_seed, crossover_rate=0.5):
    """
    Perform crossover between two individuals.
    This works by swapping some parameters of the two fuzzy neural networks.
    The parameters we consider are parent.neuron_weights, parent.V and gaussian parameters (parent.mf_params).
    crossover_rate: float, probability of swapping each parameter
    """
    child = copy_individual(parent1)
    local_parent_2 = copy_individual(parent2) #Altrimenti condivide lo stesso indirizzo di memoria da qui in avanti

    # Swap neuron weights
    if child.update_gene == "neuron_weights":
        for i in range(len(child.neuron_weights)):
            if rng_seed.random() < crossover_rate:
                child.neuron_weights[i] = local_parent_2.neuron_weights[i]

    # Swap last layer weights
    if child.update_gene == "V":
        if rng_seed.random() < crossover_rate:
            child.V = local_parent_2.V
    
    # Swap gaussian parameters
    """
    if child.update_gene == "mf_params":
        for feature_index in range(len(child.mf_params)):
            if SWAP4ALL_MFS:
                if rng_seed.random() < crossover_rate:
                    child.mf_params[feature_index] = local_parent_2.mf_params[feature_index]
            else:
                for mf_index in range(child.num_mfs):
                    if rng_seed.random() < crossover_rate:
                        child.mf_params[feature_index]["centers"][mf_index] = local_parent_2.mf_params[feature_index]["centers"][mf_index]
                        child.mf_params[feature_index]["sigmas"][mf_index] = local_parent_2.mf_params[feature_index]["sigmas"][mf_index]
    """
    
    return child


def copy_individual(individual):
    """
    Create a deep copy of an individual.
    """
    new_individual = copy.deepcopy(individual)
    new_individual.fitness = None
    return new_individual
