# Twitter Sentiment Analysis using Machine Learning with Python

## Project Overview
This project focuses on building a machine learning model to perform sentiment analysis on Twitter data. Sentiment analysis is the process of determining the emotional tone behind a piece of text. In this case, the model classifies tweets as either positive or negative.

## Dataset
The dataset used for this project is 'Sentiment140', obtained from Kaggle. It contains 1.6 million tweets extracted using the Twitter API. The tweets are labeled with a polarity (0 for negative, 4 for positive).

## Key Steps and Methodology

### 1. Data Acquisition
- The Kaggle API was used to download the `sentiment140` dataset directly into the notebook environment.
- The dataset file `training.1600000.processed.noemoticon.csv` was loaded into a pandas DataFrame.

### 2. Data Preprocessing
- **Column Renaming**: The dataset columns were assigned meaningful names: `target`, `id`, `date`, `flag`, `user`, and `text`.
- **Target Transformation**: The original 'target' column, which had values 0 (negative) and 4 (positive), was transformed so that 4 became 1, simplifying the sentiment categories to 0 (negative) and 1 (positive).
- **Missing Values**: Checked for and confirmed no missing values in the dataset.
- **Text Cleaning and Stemming**: A `steaming` function was created to:
    - Remove non-alphabetic characters.
    - Convert text to lowercase.
    - Remove stopwords (common words that don't add significant meaning).
    - Apply Porter Stemmer to reduce words to their root form (e.g., 'actor', 'actress', 'acting' all become 'act').
- A new column `steammed_content` was added to the DataFrame, storing the cleaned and stemmed text.

### 3. Feature Extraction
- The `steammed_content` was converted into numerical features using `TfidfVectorizer` (Term Frequency-Inverse Document Frequency).
- This technique converts text into a matrix of TF-IDF features, which represent the importance of words in the context of the entire dataset.

### 4. Data Splitting
- The dataset was split into training and testing sets using `train_test_split` from `sklearn`.
- 80% of the data was used for training, and 20% for testing, with `stratify=Y` to ensure an even distribution of sentiment classes in both sets.

### 5. Model Training
- A Logistic Regression model was chosen for classification.
- The model was trained on the `X_train` (TF-IDF features of stemmed content) and `Y_train` (sentiment labels).

### 6. Model Evaluation
- The trained model's performance was evaluated using `accuracy_score`.
- **Training Data Accuracy**: The model achieved an accuracy of approximately **79.87%** on the training set.
- **Test Data Accuracy**: The model achieved an accuracy of approximately **77.67%** on the test set.

### 7. Model Saving and Prediction
- The trained Logistic Regression model was saved to a file named `trained_model.sav` using Python's `pickle` module.
- The saved model can be loaded for future predictions without retraining.
- Demonstrations were provided to show how to use the loaded model to predict the sentiment of new unseen tweets.

## Technologies Used
- Python
- Pandas (for data manipulation)
- NumPy (for numerical operations)
- NLTK (for natural language processing tasks like stopwords and stemming)
- Scikit-learn (for TF-IDF vectorization, model training, and evaluation)
- Kaggle API (for dataset download)

## Conclusion
This project successfully demonstrates the process of building a sentiment analysis model for Twitter data, from data acquisition and preprocessing to model training and evaluation. The Logistic Regression model achieved a reasonable accuracy of 77.67% on unseen data, indicating its ability to classify tweet sentiments effectively.
