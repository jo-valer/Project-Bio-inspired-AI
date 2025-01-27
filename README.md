# Project-Bio-inspired-AI
Repo of our group project for the course of Bio-inspired Artificial Intelligence on _Evolutionary-Neuro-Fuzzy System for Medical Diagnosis_
## Project structure
```
Project-Bio-Inspired-AI
├── data
│   ├── datasets
│   │   ├── sepsis
│   │   │   ├── sepsis_survival_primary_cohort.csv
│   │   │   ├── sepsis_survival_study_cohort.csv
│   │   │   └── sepsis_survival_validation_cohort.csv
│   │   ├── diabetes.csv
│   │   ├── maternal_health_risk.csv
│   │   └──  obesity.csv
│   └── data.py
├── experiments
│   ├── configurations
│   │   ├── diabetes
│   │   │   ├── conf_V.json
│   │   │   ├── conf_standard.json
│   │   │   └── conf_w.json
│   │   ├── maternal_hr
│   │   │   ├── conf-00-fast.json
│   │   │   ├── conf_V.json
│   │   │   ├── conf_standard.json
│   │   │   └── conf_w.json
│   │   ├── obesity
│   │   │   ├── conf_V.json
│   │   │   ├── conf_standard.json
│   │   │   └── conf_w.json
│   │   └── sepsis
│   │   │   ├── conf-01-V.json
│   │   │   ├── conf-01-weights.json
│   │   │   └── conf-01.json
│   │   ├── conf_general_V.json
│   │   ├── conf_general_weights.json
│   │   └──  configurations.py
│   ├── results
│   │   ├── tables
│   │   │   ├── summary_results_diabetes.csv
│   │   │   ├── summary_results_maternal.csv
│   │   │   ├── summary_results_sepsis.csv
│   │   │   └── table_confrontation.py
│   │   ├── no_evo_diabete.css
│   │   ├── no_evo_maternal.csv
│   │   ├── res_w_diabetes.csv
│   │   ├── res_w_maternal.csv
│   │   ├── res_w_sepsis.csv
│   │   ├── show_results.ipynb
│   │   ├── show_results_diabets.ipynb
│   │   └── show_results_maternal.ipynb
│   ├── calculate.py
│   ├── evolution.py
│   ├── plots.py
│   └── utils.py
├── models
├── env2.yml
├── environment.yml
└── README.md
```

## Running the Code

### 1. Set up the Project Environment

- First, ensure you have Conda installed. If not, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Then, create the project environment using the provided environment.yml file (if there are some problems try with env2.yml):
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
