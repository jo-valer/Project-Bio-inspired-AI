import numpy as np


def probabilist_sum_tconorm(a, b):
    """
        Calculates the probabilistic sum t-conorm for two fuzzy values.

        Parameters:
            - a, b (float): Fuzzy values to be combined.

        Return:
            - prob_sum (float): The result of the probabilistic sum t-conorm.
    """
    prob_sum = a + b - a * b

    return prob_sum


def product_tnorm(a, b):
    """
        Calculates the product t-norm for two fuzzy values.

        Parameters:
            - a, b (float): Fuzzy values to be multiplied.

        Return:
            - prod_tnorm (float): The result of the product t-norm.
    """
    prod_tnorm = a * b

    return prod_tnorm


def andneuron(fuzzy_output, weights, interpretation="prod-probsum"):
    """
       Implements an AND neuron using specified t-norm and t-conorm operations.

       Parameters:
           - fuzzy_output (np.ndarray): The fuzzy outputs from the logical output layer.
           - weights (np.ndarray): The weights applied to the fuzzy outputs.
           - interpretation (str): A string defining the type of t-norm and t-conorm to apply (e.g., "prod-probsum").

       Return:
           - andneuron_output (float): The output of the AND neuron.
    """
    # Get the interpretation (i.e., the type) of t-norm and t-conorm to apply
    tnorm, tconorm = interpretation.split("-")
    tconorm_output = 0
    andneuron_output = 0

    if tconorm == "probsum":
        tconorm_output = probabilist_sum_tconorm(fuzzy_output, weights)

    if tnorm == "prod":
        andneuron_output = np.prod(tconorm_output)

    return andneuron_output


def orneuron(fuzzy_output, weights, interpretation="probsum-prod"):
    """
        Implements an OR neuron using specified t-norm and t-conorm operations.

        Parameters:
            - fuzzy_output (np.ndarray): The fuzzy outputs from the logical output layer.
            - weights (np.ndarray): The weights applied to the fuzzy outputs.
            - interpretation (str): A string defining the t-norm and t-conorm (e.g., "probsum-prod").

        Return:
            - orneuron_output (float): The output of the OR neuron.
    """
    tconorm, tnorm = interpretation.split("-")
    tnorm_output = 0
    orneuron_output = 0

    if tnorm == "prod":
        tnorm_output = product_tnorm(fuzzy_output, weights)

    if tconorm == "probsum":
        for v in tnorm_output:
            orneuron_output = probabilist_sum_tconorm(orneuron_output, v)

    return orneuron_output