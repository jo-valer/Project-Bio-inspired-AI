import argparse
import os 

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data.data import get_data
from experiments.configurations.configurations import get_configuration
from experiments.plots import plot_class_confusion_matrix
from models.models import FNNModel
from models.selection import selection
import copy
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import List

def initialize_population(pop_size, num_mfs, update_gene, neuron_type, fuzzy_interpretation, activation, optimizer, x_train, y_train, mutation_ind_rate, data_encoding, rng_seed):
    population = []
    
    for _ in range(pop_size):
        individuo = FNNModel(num_mfs=num_mfs, update_gene=update_gene, neuron_type=neuron_type, interpretation=fuzzy_interpretation, activation=activation, optimizer=optimizer, visualizeMF=False, mutation_ind_rate=mutation_ind_rate, data_encoding=data_encoding, rng_seed=rng_seed)
        individuo.initialize_individual(x_train, y_train)
        population.append(individuo)
        
    return population


def get_population_performance(fitness_population):
    mean_fitness = np.mean(fitness_population)
    std_fitness = np.std(fitness_population)
    max_fitness = np.max(fitness_population)
    min_fitness = np.min(fitness_population)
    
    return mean_fitness, std_fitness, max_fitness, min_fitness


def get_train_eval_split(data, percentage_train=0.8):
    x_data, y_data = data[0], data[1]
    x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, train_size=percentage_train, random_state=0, stratify=y_data)
    
    return (x_train, y_train), (x_eval, y_eval)


def run_experiment(
    train_data,
    test_data,
    data_encoding,
    pred_method,
    fitness_function,
    mutation_rate,
    mutation_ind_rate,
    crossover_rate,
    max_gen,
    max_patience,
    mu, 
    lambda_,
    selection_strategy,
    map_class_dict,
    neuron_type,
    num_mfs,
    update_gene,
    activation,
    optimizer,
    i_seed,
    rng_seed,
    local_results,
    path_to_results,
):
                        
    """
    Run an experiment with the given configuration and save the results.
    Parameters:
        - train_data (tuple): Tuple containing features and labels for the training set.
        - test_data (tuple): Tuple containing features and labels for the testing set.
        - data_encoding (str): Specifies the data encoding method to be used. Example: 'no-encoding', 'one-hot-encoding'.
        - pred_method (str): The prediction method to be used by the FNN model (e.g., argmax).
        - fitness_function (str): The fitness function to use for evaluating the FNN model.
        - mutation_rate (float): Strength of mutation.
        - mutation_ind_rate (float): Probability of mutation of an individual.
        - crossover_rate (float): Probability of crossover.
        - max_gen (int): Maximum number of generations.
        - max_patience (int): Maximum number of generations without improvement.
        - mu (int): Number of individuals to select from the population.
        - lambda_ (int): Number of individuals to generate.
        - selection_strategy (str): Selection strategy to use.
        - map_class_dict (dict): A dictionary that maps predicted class values to original dataset class values.
        - neuron_type (str): Type of neuron to use in the FNN model.
        - num_mfs (int): Number of membership functions for each input dimension.
        - update_gene (str): Type of gene update strategy.
        - activation (str): Activation function to use in the FNN model.
        - optimizer (str): Optimizer algorithm to use for training the FNN model.
        - i_seed (int): Seed for the experiment.
        - rng_seed (int): Seed for random number generation in the FNN model.
        - local_results (DataFrame): DataFrame to store the results of each experiment.
        - path_to_results (str): Path to the directory where experiment results will be saved.

    Returns:
        - result_train (float): The fitness value of the best individual on the training set.
        - result_eval (float): The fitness value of the best individual on the evaluation set.
        - result_test (float): The fitness value of the best individual on the testing set.
        - local_results (DataFrame): DataFrame containing the results of the current experiment, each row is a new generation.
    """

    current_neuron_type, fuzzy_interpretation = neuron_type.split("_")

    cm_name = f"seed_{i_seed}_neurontype_{neuron_type}_nummfs_{num_mfs}_mutrate_{mutation_rate}_mutindrate_{mutation_ind_rate}_crossrate_{crossover_rate}_maxgen_{max_gen}_maxpat_{max_patience}_mu_{mu}_lambda_{lambda_}_selstr_{selection_strategy}"
    """ if not os.path.exists(path_to_exp_results):
        os.makedirs(path_to_exp_results, exist_ok=True) """

    x_test, y_test = test_data[0], test_data[1]

    (x_train, y_train), (x_eval, y_eval) = get_train_eval_split(train_data, percentage_train=0.8)
    
    
    population:List[FNNModel] = initialize_population(mu, num_mfs, update_gene, current_neuron_type, fuzzy_interpretation, activation, 
                                       optimizer, x_train, y_train, mutation_ind_rate, data_encoding, rng_seed)
    
    generation = 0
    best_fitness = 0
    best_guy = None
    patience = max_patience
    epoch_improoved = False

    for generation in tqdm(range(max_gen),desc="Generations",position=0):
        if patience == 0:
            break
        
        population = selection(population, mu, lambda_, mutation_rate, crossover_rate, rng_seed,
                                        fitness_function, x_train, y_train, data_encoding, pred_method,
                                        map_class_dict, selection_strategy)
        
        epoch_improoved = False
        
        performance_train = []
        performance_eval = []
        performance_test = []
        
        for individuo in population:
            fitness_train = individuo.fitness
            performance_train.append(fitness_train)
            
            fitness_eval = individuo.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False)[fitness_function]
            performance_eval.append(fitness_eval)
            
            fitness_test = individuo.calculate_fitness(fitness_function, x_test, y_test, data_encoding, pred_method, map_class_dict, update_fitness = False)[fitness_function]
            performance_test.append(fitness_test)
        
            if fitness_eval > best_fitness: #Save the best individuo
                best_fitness = fitness_eval
                best_guy = copy.deepcopy(individuo)
                epoch_improoved = True

        mean_performance_train, std_performance_train, max_performance_train, min_performance_train = get_population_performance(performance_train)
        mean_performance_eval, std_performance_eval, max_performance_eval, min_performance_eval = get_population_performance(performance_eval)
        mean_performance_test, std_performance_test, max_performance_test, min_performance_test = get_population_performance(performance_test)

        new_row = pd.DataFrame({"Epoch": [generation], "Train_max_fitness": [max_performance_train], "Train_min_fitness": [min_performance_train], "Train_avg_fitness": [mean_performance_train], "Train_std_fitness": [std_performance_train], "Dev_max_fitness": [max_performance_eval], "Dev_min_fitness": [min_performance_eval], "Dev_avg_fitness": [mean_performance_eval], "Dev_std_fitness": [std_performance_eval], "Test_max_fitness": [max_performance_test], "Test_min_fitness": [min_performance_test], "Test_avg_fitness": [mean_performance_test], "Test_std_fitness": [std_performance_test]})
        local_results = pd.concat([local_results, new_row], ignore_index=True)

        if epoch_improoved:
            patience = max_patience
        else:
            patience -= 1
        
        generation += 1
    
    metrics_train = best_guy.calculate_fitness(fitness_function, x_train, y_train, data_encoding, pred_method, map_class_dict, update_fitness = False)
    metrics_eval = best_guy.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False)
    metrics_test = best_guy.calculate_fitness(fitness_function, x_test, y_test, data_encoding, pred_method, map_class_dict, update_fitness = False)
    
    fitness_train = metrics_train[fitness_function]
    fitness_eval = metrics_eval[fitness_function]
    fitness_test = metrics_test[fitness_function]

    cm_train = metrics_train["cm"]
    cm_test = metrics_test["cm"]
    cm_eval = metrics_eval["cm"]

    # plot_class_confusion_matrix(split=f"{cm_name}_TRAIN", cm = cm_train, labels= metrics_train['unique_labels'], path_to_exp_results=path_to_results)
    # plot_class_confusion_matrix(split=f"{cm_name}_TEST", cm = cm_test, labels= metrics_test['unique_labels'], path_to_exp_results=path_to_results)
    # plot_class_confusion_matrix(split=f"{cm_name}_EVAL", cm = cm_eval, labels= metrics_eval['unique_labels'], path_to_exp_results=path_to_results)

    return fitness_train, fitness_eval, fitness_test, local_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dataset", type=str, default="mammography", help="specify the dataset to use"
    )
    parser.add_argument(
        "-path_to_conf",
        type=str,
        default="./experiments/configurations/mammography/conf-01-all-evo.json",
        help="configuration file for the current experiment",
    )
    parser.add_argument(
        "-path_to_results",
        type=str,
        default="./experiments/results/",
        help="directory where to store the results",
    )

    args = parser.parse_args()

    dataset = args.dataset
    path_to_conf = args.path_to_conf
    path_to_results = args.path_to_results
    conf = get_configuration(path_to_conf)

    # experiment setting
    num_seeds = conf["num_seeds"]
    neuron_types= conf["neuron_types"]
    num_mfs_options = conf["num_mfs_options"]
    activation = conf["activation"]
    optimizer = conf["optimizer"]
    data_encoding = conf["data_encoding"]
    pred_method = conf["pred_method"]
    fitness_function = conf["fitness_function"]
    mutation_rate = conf["mutation_rate"]
    crossover_rate = conf["crossover_rate"]
    max_generations = conf["max_generations"]
    max_patience = conf["max_patience"]
    mu_values = conf["mu"]
    lambda_values = conf["lambda_"]
    selection_strategies = conf["selection_strategies"]
    mutation_individual_rate = conf["mutation_individual_rate"]
    default_path_results = path_to_results + conf["path_to_results"]
    update_genes = conf["update_genes"]

    os.makedirs(default_path_results, exist_ok=True)

    data_train, data_test, map_class_dict = get_data(dataset, data_encoding)
    
    global_results = pd.DataFrame(columns=["Seed", "NeuronType", "MFs", "update_gene", "mutation_rate", "mutation_individual_rate", "crossover_rate",  "max_generations", "max_patience", "mu", "lambda", "selection_strategy", "Train_Acc.", "Dev_Acc.", "Test_Acc.", "time"])

    # get the filename for the global results
    base_filename = "global_result_"
    extension= ".csv"
    new_file = False
    complete_filename = ""
    counter = 0
    
    while not new_file: #Check if the file already exists, if so, generate a new filename
        id = str(counter)
        complete_filename = base_filename + id + extension
        
        if not os.path.exists(f"{default_path_results}{complete_filename}"):
            new_file = True
            
        counter+=1
        
    for i_seed in range(3, num_seeds):
        rng_seed = np.random.default_rng(i_seed)
        for neuron_type in neuron_types:
            for num_mfs in num_mfs_options:
                for update_gene in update_genes:
                    for mut_rate in mutation_rate:
                        for mut_ind_rate in mutation_individual_rate:
                            for cross_rate in crossover_rate:
                                for max_gen in max_generations:
                                    for max_pat in max_patience:
                                        if max_pat > max_gen:
                                            continue
                                        for mu in mu_values:
                                            for lamb in lambda_values:
                                                if mu > lamb:
                                                    continue
                                                for sel_str in selection_strategies:
                                                    
                                                    local_results = pd.DataFrame(columns=["Epoch", "Train_max_fitness", "Train_min_fitness", "Train_avg_fitness", "Train_std_fitness", "Dev_max_fitness", "Dev_min_fitness", "Dev_avg_fitness", "Dev_std_fitness", "Test_max_fitness", "Test_min_fitness", "Test_avg_fitness", "Test_std_fitness"])
                                                    
                                                    start_time = time.time()
                                                    result_train, result_eval, result_test, local_results = run_experiment(
                                                        data_train,
                                                        data_test,
                                                        data_encoding,
                                                        pred_method,
                                                        fitness_function,
                                                        mut_rate,
                                                        mut_ind_rate,
                                                        cross_rate,
                                                        max_gen, 
                                                        max_pat,
                                                        mu, 
                                                        lamb,
                                                        sel_str,
                                                        map_class_dict,
                                                        neuron_type,
                                                        num_mfs,
                                                        update_gene,
                                                        activation,
                                                        optimizer,
                                                        i_seed,
                                                        rng_seed,
                                                        local_results,
                                                        default_path_results
                                                    )
                                                    end_time = time.time()
                                                    elapsed_time = end_time - start_time
                                                    
                                                    os.makedirs(default_path_results, exist_ok=True)
                                                    local_results.to_csv(default_path_results + f"local_results_seed_{i_seed}_neurontype_{neuron_type}_nummfs_{num_mfs}_mutrate_{mut_rate}_mutindrate_{mut_ind_rate}_crossrate_{cross_rate}_maxgen_{max_gen}_maxpat_{max_pat}_mu_{mu}_lambda_{lamb}_selstr_{sel_str}.csv")
                                                    
                                                    
                                                    new_result = pd.DataFrame({"Seed": [i_seed], "NeuronType": [neuron_type], "MFs": [num_mfs], "mutation_rate": [mut_rate], "mutation_individual_rate": [mut_ind_rate], "crossover_rate": [cross_rate],  "max_generations": [max_gen], "max_patience": [max_pat], "mu": [mu], "lambda": [lamb], "selection_strategy": [sel_str], "Train_Acc.": [result_train], "Dev_Acc.": [result_eval], "Test_Acc.": [result_test], "time": [elapsed_time]})
                                                    global_results = pd.concat([global_results, new_result], ignore_index=True)
                                    
                                    global_results.to_csv(default_path_results + complete_filename, index=False)
