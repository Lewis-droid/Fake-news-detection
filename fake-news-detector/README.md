# Fake News Detector

This project is a Fake News Detector that classifies news articles as either true or false using machine learning techniques. The model is trained on datasets containing labeled news articles.

## Project Structure

- **data/**: Contains the datasets used for training and evaluation.
  - `false.kisiiuniversity.csv`: Dataset of news articles classified as false.
  - `true.kisiiuniversity.csv`: Dataset of news articles classified as true.
  
- **models/**: Contains the serialized model file.
  - `fake_news_model.pkl`: The trained fake news detection model.

- **notebooks/**: Contains Jupyter notebooks for analysis.
  - `analysis.ipynb`: Notebook for exploratory data analysis and visualization.

- **src/**: Contains source code for data processing, model training, and evaluation.
  - `data_preprocessing.py`: Functions for loading and cleaning the datasets.
  - `model_training.py`: Implementation of the model training process.
  - `model_evaluation.py`: Functions to evaluate the model's performance.
  - `utils.py`: Utility functions for data loading and saving predictions.

- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fake-news-detector
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

1. Preprocess the data using `data_preprocessing.py`.
2. Train the model using `model_training.py`.
3. Evaluate the model's performance using `model_evaluation.py`.
4. Use the Jupyter notebook `analysis.ipynb` for exploratory data analysis and visualization of the datasets.

## License

This project is licensed under the MIT License.