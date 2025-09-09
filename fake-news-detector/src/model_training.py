from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib

def load_data():
    false_data = pd.read_csv('data/false.kisiiuniversity.csv')
    true_data = pd.read_csv('data/true.kisiiuniversity.csv')
    
    false_data['label'] = 0  # 0 for false
    true_data['label'] = 1    # 1 for true
    
    data = pd.concat([false_data, true_data], ignore_index=True)
    return data['text'], data['label']

def train_model(X, y):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X, y)
    return model

def save_model(model, filename='models/fake_news_model.pkl'):
    joblib.dump(model, filename)

if __name__ == "__main__":
    X, y = load_data()
    model = train_model(X, y)
    save_model(model)