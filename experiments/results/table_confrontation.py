import csv
from collections import defaultdict

# Funzione per leggere i dati dal CSV e raggrupparli per configurazione
def read_and_group_data(file_path):
    grouped_data = defaultdict(list)

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Creazione della configurazione combinando i campi richiesti
            config = f"{row['NeuronType']}, {row['MFs']}, {row['mutation_rate']}, {row['mutation_individual_rate']}, {row['crossover_rate']}"
            grouped_data[config].append({
                "Train_Acc.": float(row["Train_Acc."] or 0),
                # "Dev_Acc.": float(row["Dev_Acc."] or 0),
                "Test_Acc.": float(row["Test_Acc."] or 0),
                "time": float(row["time"] or 0),
            })

    return grouped_data

def read_and_group_data_base(file_path):
    grouped_data = defaultdict(list)

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Creazione della configurazione combinando i campi richiesti
            config = f"{row['NeuronType']}, {row['MFs']}"
            grouped_data[config].append({
                "Train_Acc.": float(row["Train_Acc."] or 0),
                # "Dev_Acc.": float(row["Dev_Acc."] or 0),
                "Test_Acc.": float(row["Test_Acc."] or 0),
                "time": float(row["time"] or 0),
            })

    return grouped_data

# Funzione per calcolare le medie per ogni configurazione
def calculate_averages(grouped_data):
    averaged_data = []

    for config, values in grouped_data.items():
        num_entries = len(values)
        avg_train_acc = sum(v["Train_Acc."] for v in values) / num_entries
        # avg_dev_acc = sum(v["Dev_Acc."] for v in values) / num_entries
        avg_test_acc = sum(v["Test_Acc."] for v in values) / num_entries
        avg_time = sum(v["time"] for v in values) / num_entries

        #separete in the configuration the neuron type from the rest
        neuron_type = config.split(",")[0]
        #all the rest of the configuration
        config = config[len(neuron_type):]
        averaged_data.append({
            "Neuron Type": neuron_type[:-1],
            "Configuration": config[2:],
            "Train Acc.": round(avg_train_acc, 3),
            # "Dev Acc.": round(avg_dev_acc, 3),
            "Test Acc.": round(avg_test_acc, 3),
            "Tempo (s)": round(avg_time, 2),
        })

    return averaged_data

# Funzione per scrivere i dati medi in un nuovo CSV
def write_summary_to_csv(output_path, baseline_data, experiment_data):
    header = ["Neuron Type", "Configuration", "Train Acc.", "Test Acc.", "Tempo (s)"]

    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        # Scrivi la baseline
        writer.writerows(baseline_data)

        # Scrivi i dati degli esperimenti
        writer.writerows(experiment_data)

# Percorsi dei file
baseline_file = "NeuroFuzzyProject/experiments/results/sepsis/altro.csv"
experiment_file = "NeuroFuzzyProject/experiments/results/res_w_sepsis.csv"
output_file = "NeuroFuzzyProject/experiments/results/summary_results_sepsis.csv"

# Elaborazione della baseline
baseline_data = read_and_group_data_base(baseline_file)
baseline_summary = calculate_averages(baseline_data)

# Elaborazione degli esperimenti
experiment_data = read_and_group_data(experiment_file)
experiment_summary = calculate_averages(experiment_data)

# Scrittura del file finale
write_summary_to_csv(output_file, baseline_summary, experiment_summary)

print(f"Tabella riepilogativa salvata in '{output_file}'!")
