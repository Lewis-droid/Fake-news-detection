import pandas as pd

def load_data(false_data_path, true_data_path):
    false_data = pd.read_csv(false_data_path)
    true_data = pd.read_csv(true_data_path)
    return false_data, true_data

def clean_data(data):
    # Implement data cleaning steps here
    # For example: removing duplicates, handling missing values, etc.
    cleaned_data = data.drop_duplicates().dropna()
    return cleaned_data

def prepare_data(false_data, true_data):
    false_data['label'] = 0  # 0 for false news
    true_data['label'] = 1   # 1 for true news
    combined_data = pd.concat([false_data, true_data], ignore_index=True)
    return combined_data.sample(frac=1).reset_index(drop=True)  # Shuffle the data

def preprocess_data(false_data_path, true_data_path):
    false_data, true_data = load_data(false_data_path, true_data_path)
    cleaned_false_data = clean_data(false_data)
    cleaned_true_data = clean_data(true_data)
    final_data = prepare_data(cleaned_false_data, cleaned_true_data)
    return final_data