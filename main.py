import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from joblib import dump, load

# Step 1: Load SpaCy for NLP
nlp = spacy.load("en_core_web_md")

# Step 2: Load Data
def load_data(file_path, is_test=False):
    """  
        Parameters:
        file_path (str): The path to the TSV file.
        is_test (bool): Flag indicating whether the data is for testing. Defaults to False.

        Returns:
        pandas.DataFrame: The loaded data as a DataFrame.

        For training and validation data, the DataFrame will have columns: ['split', 'name', 'context'].
        For test data, the DataFrame will have columns: ['id', 'context'].

    Load data from a TSV file into a pandas DataFrame.
    For training and validation, columns: ['split', 'name', 'context']
    For test data, columns: ['id', 'context']
    """
    if not is_test:
        return pd.read_csv(file_path, sep='\t', header=None, names=['split', 'name', 'context'], on_bad_lines='skip')
    else:
        return pd.read_csv(file_path, sep='\t', header=None, names=['id', 'context'], on_bad_lines='skip')

# Step 3: Preprocess Text
def preprocess_text_spacy(text):
    """ 
    Preprocesses the input text by tokenizing it, and removing stop words and punctuation using SpaCy.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text with tokens in lowercase, and without stop words and punctuation.
    
    Preprocess text by tokenizing, removing stop words, and punctuation using SpaCy.
    """
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]
    return " ".join(tokens)

def apply_preprocessing(data, column):
    """
    Apply preprocessing to a specific column in the given DataFrame.
    This function applies the `preprocess_text_spacy` function to each element
    in the specified column of the DataFrame.
    Args:
        data (pd.DataFrame): The DataFrame containing the data to be processed.
        column (str): The name of the column to be preprocessed.
    Returns:
        pd.DataFrame: The DataFrame with the preprocessed column.
    """
    
    data[column] = data[column].apply(preprocess_text_spacy)
    return data

# Step 4: Extract Full Context (n-grams)
def extract_full_context(text):
    """
    Extracts the full context from a given text by removing the redaction marker '█' and 
    concatenating the tokens before and after the marker.
    Args:
        text (str): The input text containing a redaction marker '█'.
    Returns:
        str: The text with the redaction marker removed and the surrounding tokens concatenated.
    """
    
    if "█" in text:
        before, after = text.split("█", 1)
        before_tokens = before.split()  # Tokens before the redaction
        after_tokens = after.split()    # Tokens after the redaction
        return " ".join(before_tokens + after_tokens)
    return text

def apply_full_context_extraction(data, column):
    """
    Applies the full context extraction function to a specified column in a DataFrame.
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to apply the extraction function to.
    Returns:
    pd.DataFrame: The DataFrame with the specified column modified by the extraction function.
    """
    
    data[column] = data[column].apply(extract_full_context)
    return data

# Step 5: Grid Search for Hyperparameter Tuning
def perform_grid_search(X_train, y_train):
    """
    Perform a grid search to find the best hyperparameters for a RandomForestClassifier.
    Parameters:
    X_train (array-like or sparse matrix): The training input samples.
    y_train (array-like): The target values (class labels) as integers or strings.
    Returns:
    RandomForestClassifier: The best estimator found by the grid search.
    """
    
    param_grid = {
        'n_estimators': [50, 100],       # Number of trees in the forest
        'max_depth': [None],           # Maximum depth of the tree
        'min_samples_split': [5, 10],     # Minimum number of samples to split a node
        'min_samples_leaf': [2, 4]        # Minimum number of samples in a leaf node
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Step 6: Evaluate Model Performance
def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a classification model using precision, recall, and F1-score.
    Parameters:
    y_true (array-like): True labels of the dataset.
    y_pred (array-like): Predicted labels by the model.
    Prints:
    - Classification report including precision, recall, and F1-score for each class.
    - Overall precision, recall, and F1-score (macro average).
    """
   
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print("Model Evaluation:")
    #print(classification_report(y_true, y_pred))
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# Step 7: Main Workflow
def main():
    """
    Main function to execute the machine learning pipeline for name unredaction.
    This function performs the following steps:
    1. Loads and preprocesses training and validation data.
    2. Splits the data into training and validation sets.
    3. Initializes a TF-IDF vectorizer and transforms the data.
    4. Performs grid search to find the best model.
    5. Saves the best model and vectorizer.
    6. Evaluates the model on validation data.
    7. Loads and preprocesses test data.
    8. Transforms the test data and makes predictions.
    9. Saves the predictions to a file.
    The function assumes the existence of specific helper functions:
    - load_data
    - apply_preprocessing
    - apply_full_context_extraction
    - perform_grid_search
    - evaluate_model
    The function also assumes the presence of the following files:
    - 'data/unredactor.tsv' for training data
    - 'data/test.tsv' for test data
    The predictions are saved to 'submission.tsv'.
    """
    # File paths
    train_file_path = 'data/unredactor.tsv'
    test_file_path = 'data/test.tsv'

    # Load and preprocess training and validation data
    #print("Loading training data...")
    data = load_data(train_file_path)
    data = apply_preprocessing(data, 'context')
    data = apply_full_context_extraction(data, 'context')

    # Split into training and validation sets
    train_data = data[data['split'] == 'training']
    validation_data = data[data['split'] == 'validation']

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=10000,ngram_range=(1, 3), stop_words='english')

    # Transform training and validation data
    #print("Transforming data...")
    X_train = vectorizer.fit_transform(train_data['context'])
    y_train = train_data['name']
    X_validation = vectorizer.transform(validation_data['context'])
    y_validation = validation_data['name']

    # Perform grid search and train the model
    #print("Performing grid search...")
    best_model = perform_grid_search(X_train, y_train)

    # Save the best model and vectorizer
    dump(best_model, 'best_random_forest.joblib')
    dump(vectorizer, 'tfidf_vectorizer.joblib')

    # Predict on validation data
    #print("Evaluating model on validation data...")
    y_pred = best_model.predict(X_validation)
    evaluate_model(y_validation, y_pred)

    # Load and preprocess test data
    #print("Loading and preprocessing test data...")
    test_data = load_data(test_file_path, is_test=True)
    test_data = apply_preprocessing(test_data, 'context')
    test_data = apply_full_context_extraction(test_data, 'context')

    # Transform test data
    X_test = vectorizer.transform(test_data['context'])

    # Predict on test data
    predicted_names = best_model.predict(X_test)

    # Save predictions to a file
    predictions_df = pd.DataFrame({
        'id': test_data['id'],
        'name': predicted_names
    })
    predictions_df.to_csv('submission.tsv', sep='\t', index=False)
    print("\nPredictions saved to 'submission.tsv'.")

# Execute main
if __name__ == "__main__":
    main()
