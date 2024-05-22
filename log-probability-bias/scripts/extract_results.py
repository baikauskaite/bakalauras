import os
import pandas as pd

# Example usage
directory_path = '../results/8_run/gottbert-base-debiased-finetuned-mlm'

def extract_values_from_file(file_path):
    effect_size = None
    p_value = None

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Effect size" in line:
                effect_size = float(line.split(',')[1].strip())
            elif "Permutation test p-value" in line:
                p_value = float(line.split(',')[1].strip())
    
    return effect_size, p_value

def process_files(directory_path):
    data = []
    file_names = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith('significance.txt'):
            file_path = os.path.join(directory_path, file_name)
            effect_size, p_value = extract_values_from_file(file_path)
            base_name = os.path.splitext(file_name)[0]
            file_names.append(base_name)
            data.append((effect_size, p_value))
    
    df = pd.DataFrame(data, columns=['Effect Size', 'Permutation Test p-value'], index=file_names)
    return df

df = process_files(directory_path)

print(df)
