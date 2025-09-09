from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='true')
    recall = recall_score(y_test, y_pred, pos_label='true')
    f1 = f1_score(y_test, y_pred, pos_label='true')

    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return evaluation_results