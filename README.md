# Project-Bio-inspired-AI
Repo of our group project for the course of Bio-inspired Artificial Intelligence on _Evolutionary-Neuro-Fuzzy System for Medical Diagnosis_

## Running the Code

### 1. Set up the Project Environment

- First, ensure you have Conda installed. If not, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Then, create the project environment using the provided environment.yml file:
  ```bash
  conda env create -f environment.yml
  ```

- And activate it:

  ```bash
  conda activate neurofuzzy
  ```

### 2. Set up Configuration

- Ensure you have a configuration file located in the `experiments/configurations/<dataset>/` directory.

- This file should contain experiment settings such as the number of seeds, neuron types, fitness fn, parameters for mutation and crossover etc, and the path for storing the results.

### 3. Command-line Arguments

- The script accepts command-line arguments to a specific dataset and path to the configuration file

- Use the following command-line arguments:
  - `-dataset`: Specify the dataset to use
  - `-path_to_conf`: Provide a path to the configuration file

### 4. Run the Script

- Run the main script using Python:
  ```bash
  python main.py -dataset <dataset> -path_to_conf ./experiments/configurations/<dataset>/<name_of_conf>.json 
  ```
