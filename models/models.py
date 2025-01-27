
import numpy as np
import copy

from numpy.linalg import pinv
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import (
    AUC,
    Precision,
    Recall,
    SensitivityAtSpecificity,
    SpecificityAtSensitivity,
)

from models.operators import andneuron, orneuron
from experiments.plots import visualize_membership_functions


class FNNModel:
    def __init__(
        self,
        num_mfs,
        update_gene="all",
        neuron_type="andneuron",
        interpretation="prod-probsum",
        activation="linear",
        optimizer="moore-penrose",
        visualizeMF=False,
        mutation_ind_rate = 0.5,
        data_encoding = "one-hot-encoding",
        rng_seed=None
    ):
        """
        Initialize a Fuzzy Neural Network (FNN) model with the specified configuration.

        Parameters:
            - num_mfs (int): Number of membership functions for each input dimension.
            - neuron_type (str): Type of neuron to use in the FNN model. Default is "andneuron".
            - fuzzy_interpretation: String denoting the fuzzy interpretation to apply when computing the output of the
              logical neurons (i.e., AND and OR). Default is "prod-probsum"
            - activation (str): Activation function to use in the FNN model. Default is "linear".
            - optimizer (str): Optimizer algorithm to use for training the FNN model. Default is "moore-penrose".
            - visualizeMF (bool): Whether to visualize membership functions during training. Default is False.
            - rng_seed (int or None): Seed for random number generation. If None, a default RNG is used. Default is None.

        Returns:
            None
        """

        self.model = None
        self.num_mfs = num_mfs
        self.neuron_type = neuron_type
        self.intepretation = interpretation
        self.activation = activation
        self.optimizer = optimizer
        self.visualizeMF = visualizeMF
        self.data_encoding = data_encoding

        if rng_seed is None:
            self.rng_seed = np.random.default_rng(0)
        else:
            self.rng_seed = rng_seed

        self.mf_params = []
        self.neuron_weights = None
        self.total_fuzzy_neurons = None
        self.rules_dictionary = []  # Stores fuzzy rules
        self.axioms = []  # Stores axioms generated from fuzzy rules
        self.fitness = None
        self.mutation_ind_rate = mutation_ind_rate # Probability for the individual to mutate 
        self.update_gene = update_gene
        
        self.fuzzy_outputs = None
        
    def initialize_individual(self, x_train, y_train):
        fuzzy_outputs = self.fuzzification_layer(x_train)
        self.fuzzy_outputs = copy.deepcopy(fuzzy_outputs)
        logic_outputs = self.logic_neurons_layer(fuzzy_outputs)

        if self.data_encoding not in ["one-hot-encoding", "no-encoding"]:
            raise ValueError("Invalid data encoding method")
        
        if self.optimizer not in ["moore-penrose"]:
            raise ValueError("Invalid optimizer method")
        
        if self.optimizer == "moore-penrose":
            self.V = np.dot(pinv(logic_outputs), y_train)

    def calculate_fitness(self, fitness_type, x, y, data_encoding, pred_method, map_class_dict, update_fitness=True, fast = True):
        evaluation_metrics_train = self.evaluate_model(x, y, data_encoding, pred_method, map_class_dict,fast=fast)
        
        fitness_value = evaluation_metrics_train[fitness_type]
        
        if update_fitness: # True only on the train set
            self.fitness = fitness_value
        
        return evaluation_metrics_train
    
    def generate_parameters(self, x_train, y_train):
        if self.update_gene != "V":
            #fuzzy_outputs = self.fuzzification_layer(x_train)
            logic_outputs = self.logic_neurons_layer(copy.deepcopy(self.fuzzy_outputs))
            self.V = np.dot(pinv(logic_outputs), y_train)
        
        self.fitness = None #Devi rimuovere il fitness perche' cambiano i parametri
    
    def mutate(self, mutation_rate=0.1):
        """
        Mutate the individual (V) by adding random noise to the weights.

        Parameters:
            - mutation_rate (float): Mutation rate, which determines the amount of noise to add to the weights.

        Returns:
            None
        """
        random_number = self.rng_seed.uniform()
        if random_number < self.mutation_ind_rate:
            # Reset fitness
            self.fitness = None

            if self.update_gene == "V":
                self.V += self.rng_seed.normal(0, mutation_rate, self.V.shape)

            elif self.update_gene == "neuron_weights":
                self.neuron_weights += self.rng_seed.normal(0, mutation_rate, self.neuron_weights.shape)

                #The weights should be between 0 and 1:
                self.neuron_weights[self.neuron_weights < 0] = 0
                self.neuron_weights[self.neuron_weights > 1] = 1

                """elif self.update_gene == "mf_params":
                    for feature_index in range(len(self.mf_params)):
                        for mf_index in range(self.num_mfs):
                            self.mf_params[feature_index]["centers"][mf_index] += self.rng_seed.normal(0, mutation_rate)
                            self.mf_params[feature_index]["sigmas"][mf_index] += self.rng_seed.normal(0, mutation_rate)
                            if self.mf_params[feature_index]["sigmas"][mf_index] < 0:
                                self.mf_params[feature_index]["sigmas"][mf_index] = 0.1"""

            else:
                raise ValueError("Invalid update_gene method")



    def fuzzification_layer(self, x):
        """
        Performs the fuzzification layer operation for the input data.

        Parameters:
            - x (numpy.ndarray): Input data of shape (num_samples, num_features).

        Returns:
            - fuzzy_outputs (numpy.ndarray): Fuzzy outputs after fuzzification, with shape (num_samples, total_fuzzy_neurons).
              Each row represents the fuzzy output for a sample, and each column represents the degree of membership to a fuzzy set.
        """

        num_samples, num_features = x.shape
        if self.total_fuzzy_neurons is None:
            self.total_fuzzy_neurons = self.num_mfs**num_features

        fuzzy_outputs = np.zeros((num_samples, self.total_fuzzy_neurons))
        # If the parameters have not been created yet
        if len(self.mf_params) == 0:
            self.total_fuzzy_neurons = self.num_mfs**num_features
            self.mf_params = [
                {"centers": [], "sigmas": []} for _ in range(num_features)
            ]
            self.neuron_weights = self.rng_seed.random(
                (self.total_fuzzy_neurons, num_features)
            )  # Generates random weights

            for feature_index in range(num_features):
                feature_min, feature_max = np.min(x[:, feature_index]), np.max(
                    x[:, feature_index]
                )
                centers = np.linspace(feature_min, feature_max, self.num_mfs)
                sigmas = np.full_like(
                    centers,
                    (feature_max - feature_min) / (self.num_mfs * np.sqrt(2 * np.pi)),
                )

                self.mf_params[feature_index] = {"centers": centers, "sigmas": sigmas}
                # self.rules_dictionary.append({"centers": centers, "sigmas": sigmas})

                if self.visualizeMF:
                    visualize_membership_functions(feature_index, centers, sigmas)

        # Get centers and sigmas, and compute fuzzy outputs
        for feature_index in range(num_features):
            for mf_index in range(self.num_mfs):
                sigma = self.mf_params[feature_index]["sigmas"][mf_index]
                center = self.mf_params[feature_index]["centers"][mf_index]
                gaussian_output = norm.pdf(x[:, feature_index], center, sigma)
                fuzzy_outputs[:, feature_index * self.num_mfs + mf_index] = (
                    gaussian_output
                )
                if 'nan' in fuzzy_outputs or np.isnan(fuzzy_outputs).any():
                    print("Nan in fuzzy output")

        self.rules_dictionary = self.generate_rules_dictionary()
        return fuzzy_outputs  # Return the fuzzy outputs

    def generate_rules_dictionary(self):
        """
            Generates a dictionary of rules based on combinations of MFs for all characteristics.
        """
        rules_dictionary = []
        num_features = len(self.mf_params)
        num_mfs_per_feature = self.num_mfs
        total_rules = num_mfs_per_feature**num_features

        for rule_index in range(total_rules):
            mf_combination = np.unravel_index(
                rule_index, [num_mfs_per_feature] * num_features
            )
            rule_params = {"centers": [], "sigmas": []}
            for feature_index, mf_index in enumerate(mf_combination):
                rule_params["centers"].append(
                    self.mf_params[feature_index]["centers"][mf_index]
                )
                rule_params["sigmas"].append(
                    self.mf_params[feature_index]["sigmas"][mf_index]
                )
            rules_dictionary.append(rule_params)

        return rules_dictionary

    def logic_neurons_layer(self, fuzzy_outputs):
        """
        Calculates the logical output of the fuzzy neural network based on the fuzzy outputs and neuron weights.

        Parameters:
            - fuzzy_outputs (numpy.ndarray): Fuzzy outputs after fuzzification, with shape (num_samples, total_fuzzy_neurons).
              Each row represents the fuzzy output for a sample, and each column represents the degree of membership to a fuzzy set.
        Returns:
            - logic_output (numpy.ndarray): Logical output of the fuzzy neural network, with shape (num_samples, total_fuzzy_neurons).
              Each row represents the logical output for a sample, and each column represents the output of a logical neuron
              corresponding to a specific combination of membership functions for input features.
        """

        num_samples = fuzzy_outputs.shape[0]
        num_features = len(self.mf_params)
        num_mfs_per_feature = self.num_mfs
        self.total_fuzzy_neurons = num_mfs_per_feature**num_features

        # Initializes the logical output vector `z`
        z = np.zeros((num_samples, self.total_fuzzy_neurons))

        # Iterates over all possible combinations of MFs for all features
        for sample_index in range(num_samples):
            for neuron_index in range(self.total_fuzzy_neurons):
                # Obtains the specific combination of MFs for the current neuron index
                mf_combination = np.unravel_index(
                    neuron_index, [num_mfs_per_feature] * num_features
                )

                # Calculates the logical neuron output for the MF combination
                logic_neuron_input = [
                    fuzzy_outputs[
                        sample_index, feature_index * num_mfs_per_feature + mf_index
                    ]
                    for feature_index, mf_index in enumerate(mf_combination)
                ]
                weights = self.neuron_weights[neuron_index]

                if self.neuron_type == "andneuron":
                    z[sample_index, neuron_index] = andneuron(
                        logic_neuron_input, weights, self.intepretation
                    )
                elif self.neuron_type == "orneuron":
                    z[sample_index, neuron_index] = orneuron(
                        logic_neuron_input, weights, self.intepretation
                    )

        logic_outputs = z

        return logic_outputs

    def train_model(self, x_train, y_train):
        """
        Trains the fuzzy neural network model using the provided training data.

        Parameters:
            - x_train (numpy.ndarray): Input features of the training data.
            - y_train (numpy.ndarray): Target labels of the training data.

        Returns:
            None
        """

        # Adjust the trainModel method to capture the return value from fuzzificationLayer
        fuzzy_outputs = self.fuzzification_layer(x_train)  # Capture the fuzzy outputs
        logic_outputs = self.logic_neurons_layer(
            fuzzy_outputs
        )  # Pass the fuzzy outputs to the logic neurons layer

        if self.optimizer == "moore-penrose":
            # Use Moore-Penrose pseudo-inverse for training
            self.V = np.dot(pinv(logic_outputs), y_train)
        elif self.optimizer == "adam":
            # TODO: up to now this is not used
            # Use a neural network for training
            self.neural_network_layer(logic_outputs, y_train)
            last_layer_weights = self.model.layers[-1].get_weights()[0]
            # Flatten the weights to make them consistent with V format and store in VR
            self.VR = np.array((last_layer_weights.flatten()))
            self.V = last_layer_weights.flatten()

    # TODO: up to now neural network layer is not used
    def neural_network_layer(self, x, y):
        self.model = Sequential(
            [
                Dense(
                    1,
                    input_dim=x.shape[1],
                    activation="linear",
                    kernel_initializer="random_uniform",
                    bias_initializer="zeros",
                )
            ]
        )
        # Adds AUC, precision, and recall to the metrics
        self.model.compile(
            optimizer="sgd",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(name="auc"),
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )
        self.model.fit(x, y, epochs=3, batch_size=1)

    def evaluate_model(self, x_test, y_test, data_encoding, pred_method, map_class_dict, fast = True):
        """
        Evaluate the trained FNN model using test data.

        Parameters:
            - x_test (numpy.ndarray): Features of the test data.
            - y_test (numpy.ndarray): Labels of the test data.
            - data_encoding (str) : Specifies the data encoding method to be used. This parameter affects how data
              is processed within the model. Example: 'no-encoding', 'one-hot-encoding'.
            - pred_method (str): The prediction method to be used by the FNN model. Example: 'sign', which may
              indicate a particular thresholding or decision strategy for final predictions.
            - map_class_dict (dict): A dictionary that maps the predicted class values (used internally by the model)
            to their original dataset class values.

        Returns:
            evaluation_metrics (dict): Dictionary containing evaluation metrics including accuracy, specificity,
            precision, recall, and F-score.
        """
        if(fast):
            entries_considered = min(100, int(0.1 * x_test.shape[0]))
            #randomly select entries
            indices = self.rng_seed.choice(x_test.shape[0], entries_considered, replace=False)
            x_test = x_test[indices]
            y_test = y_test[indices]

            #print(f"Fast evaluation: Only {len(x_test)} entries are considered for evaluation")
        # Evaluate the trained FNN model

        evaluation_metrics = {}
        fuzzy_outputs_test = self.fuzzification_layer(x_test)
        logic_outputs_test = self.logic_neurons_layer(fuzzy_outputs_test)

        if self.optimizer == "moore-penrose":
            output_v = np.dot(logic_outputs_test, self.V)

            # TODO: no-encoding + sign was related to the previous version (we may consider to remove it),
            # the 'official' and current version is  the ones with one hot encoding + argmax
            if data_encoding == "no-encoding" and pred_method== "sign":
                y_pred = np.sign(output_v)

                # show again original classes
                y_test[y_test == -1] = map_class_dict[-1]
                y_pred[y_pred == -1] = map_class_dict[-1]

            elif data_encoding == "one-hot-encoding" and pred_method == "argmax":
                y_pred = np.argmax(output_v, 1)
                y_test = np.argmax(y_test, 1)
                if map_class_dict:
                    # show again original classes
                    y_pred[y_pred == 1] = map_class_dict[1]
                    y_pred[y_pred == 0] = map_class_dict[0]

                    y_test[y_test == 1] = map_class_dict[1]
                    y_test[y_test == 0] = map_class_dict[0]

            n_class = len(np.unique(y_test))
            cm = confusion_matrix(y_test, y_pred)

            if n_class == 2:
                average_metrics = "binary"

                # Unravel confusion matrix elements
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp)
            else:
                average_metrics = "macro"

                # For multi-class, we need to calculate specificity for each class in a one-vs-rest way
                specificities = []
                for i in range(n_class):  # Iterate over classes
                    # True positives for class i
                    tp = cm[i, i]
                    # All non-class i elements
                    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
                    # False positives for class i
                    fp = np.sum(cm[:, i]) - cm[i, i]
                    # False negatives for class i
                    fn = np.sum(cm[i, :]) - cm[i, i]

                    specificity_i = tn / (tn + fp)  # Specificity for class i
                    specificities.append(specificity_i)

                specificity = np.mean(specificities)  # Average specificity for multi-class

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average=average_metrics, zero_division=0)
            recall = recall_score(y_test, y_pred, average=average_metrics)
            f1 = f1_score(y_test, y_pred, average=average_metrics)

            evaluation_metrics["accuracy"] = round(float(accuracy), 3)
            evaluation_metrics["specificity"] = round(specificity, 3)
            evaluation_metrics["precision"] = round(precision, 3)
            evaluation_metrics["recall"] = round(recall, 3)
            evaluation_metrics["fscore"] = round(f1, 3)
            evaluation_metrics["cm"] = cm
            evaluation_metrics["unique_labels"] = np.unique(y_test)

            # Prints the metrics
            """ print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print(f"Specificity: {specificity}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F-Score: {f1}") """

        else:
            scores = self.model.evaluate(logic_outputs_test, y_test)
            #print(f"\nAccuracy: {scores[1] * 100:.2f}%")

            evaluation_metrics["accuracy"] = scores[1]

        return evaluation_metrics

    def generate_fuzzy_rules(self):
        """
        Generate fuzzy rules based on the weights of the neurons in the logic layer.

        Returns:
            list: A list of fuzzy rules, where each rule is represented as a string.
        """
        # Preparation of rule strings
        rules = []

        # For each neuron in the logic layer
        for neuron_index in range(self.total_fuzzy_neurons):
            # Obtains the specific combination of MFs for the current neuron index
            mf_combination = np.unravel_index(
                neuron_index, [self.num_mfs] * len(self.mf_params)
            )
            rule = "IF "

            # For each feature, add the corresponding MF to the rule
            for feature_index, mf_index in enumerate(mf_combination):
                weight = self.neuron_weights[
                    neuron_index, feature_index
                ]  # Access the specific weight
                rule += (
                    f"x{feature_index+1} is MF{mf_index+1} with impact {weight:.2f} "
                )
                if feature_index < len(mf_combination) - 1:
                    rule += "AND " if self.neuron_type == "andneuron" else "OR "

            rule += f"THEN output is {np.round(self.V[neuron_index], 2)}"

            # Adds the complete rule to the list of rules
            rules.append(rule)

        return rules

    # TO ME, THE FOLLOWING FUNCTION SEEMS THE SAME OF PREVIOUS ONE...
    # IF YOU HAVE TO USE IT DECOMMENT IT AND ADD DOCUMENTATION

    # def generate_fuzzy_rules_with_impact(self):
    #     rules_with_impact = []
    #
    #     # For each neuron in the logic layer
    #     for neuron_index in range(self.total_fuzzy_neurons):
    #         # Obtains the specific combination of MFs for the current neuron index
    #         mf_combination = np.unravel_index(
    #             neuron_index, [self.num_mfs] * len(self.mf_params)
    #         )
    #         rule_with_impact = "IF "
    #
    #         # For each feature, add the corresponding MF to the rule with its weight (impact)
    #         for feature_index, mf_index in enumerate(mf_combination):
    #             weight = self.neuron_weights[
    #                 neuron_index, feature_index
    #             ]  # Access the specific weight
    #             rule_with_impact += (
    #                 f"x{feature_index+1} is MF{mf_index+1} with impact {weight:.2f} "
    #             )
    #             if feature_index < len(mf_combination) - 1:
    #                 rule_with_impact += (
    #                     "AND " if self.neuron_type == "andneuron" else "OR "
    #                 )
    #
    #             # Adds the rule's result (output)
    #         rule_with_impact += f"THEN output is {self.V[neuron_index]}"
    #
    #     # Adds the complete rule to the list of rules
    #     rules_with_impact.append(rule_with_impact)
    #
    #     return rules_with_impact

    # IF YOU HAVE TO USE IT DECOMMENT IT AND ADD DOCUMENTATION, STILL HAVE SOME DOUBTS ON IT

    # def visualize_3d_logic_neuron_output(self, X1_range, X2_range, resolution=100):
    #     """
    #     Visualize 3D projection of a logical neuron output.
    #
    #     Parameters:
    #     - X1_range: Tuple (min, max) for the first input feature range.
    #     - X2_range: Tuple (min, max) for the second input feature range.
    #     - resolution: Number of points per dimension in the grid.
    #     """
    #     # Generate a grid of points within the specified ranges
    #     x1 = np.linspace(X1_range[0], X1_range[1], resolution)
    #     x2 = np.linspace(X2_range[0], X2_range[1], resolution)
    #     X1, X2 = np.meshgrid(x1, x2)
    #
    #     # Flatten the grid to pass through the fuzzification layer
    #     grid_flat = np.c_[X1.ravel(), X2.ravel()]
    #
    #     # Fuzzify the grid points
    #     fuzzy_outputs = self.fuzzification_layer(grid_flat)
    #
    #     # Compute the logical outputs
    #     self.logic_neurons_layer()
    #
    #     # Assuming self.logic_output now contains the outputs for the grid, reshape it back to the grid shape
    #     Z = self.logic_output.reshape(X1.shape)
    #
    #     # Create a 3D plot
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #
    #     # Plot the surface
    #     surf = ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none")
    #     fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to indicate the values
    #
    #     ax.set_xlabel("Input Feature 1")
    #     ax.set_ylabel("Input Feature 2")
    #     ax.set_zlabel("Logical Neuron Output")
    #     ax.set_title("3D Projection of Logical Neuron Output")
    #
    #     plt.show()

    # TODO: old part related to LTN integratin, probably we can remove it
    def generate_fuzzy_axioms(self):
        """
        Generate logical axioms from fuzzy rules, including the output weights,
        adjusted for the correct number of dimensions with specific formatting.

        Returns:
        None
        """

        self.axioms.clear()  # Clear existing axioms list

        # For each logic neuron's combination
        for neuron_index, v in enumerate(self.V):
            mf_combination = np.unravel_index(
                neuron_index, [self.num_mfs] * len(self.mf_params)
            )

            rule_parts = []  # To store individual parts of the rule

            for feature_index, mf_index in enumerate(mf_combination):
                center = self.mf_params[feature_index]["centers"][mf_index]
                sigma = self.mf_params[feature_index]["sigmas"][mf_index]

                # Generate interval around the center using sigma
                interval_start = round(center - sigma, 2)
                interval_end = round(center + sigma, 2)

                # Add rule part for the current feature with interval formatting
                rule_parts.append(
                    f"x{feature_index+1} is around [{interval_start} - {interval_end}] with sigma {sigma:.2f}"
                )

            # Join rule parts with "AND" or "OR" depending on neuron type
            conjunction = " AND " if self.neuron_type == "andneuron" else " OR "
            rule_body = conjunction.join(rule_parts)

            # Format complete axiom with output rounded to two decimal places
            axiom = f"IF {rule_body}, THEN output is [{np.round(v, 2)}]."
            self.axioms.append(axiom)
