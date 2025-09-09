def load_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_predictions(predictions, file_path):
    import pandas as pd
    predictions.to_csv(file_path, index=False)

def preprocess_text(text):
    import re
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()  # Remove leading and trailing spaces

def split_data(data, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(data, test_size=test_size, random_state=random_state)