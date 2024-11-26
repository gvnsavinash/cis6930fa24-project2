# The Unredactor

---

**Author**: Venkata Naga Satya Avinash , Gudipudi

---

**Introduction**
"The Unredactor" is an innovative machine learning application designed to identify and reconstruct redacted names within textual datasets. Leveraging advanced Natural Language Processing (NLP) techniques and the power of RandomForestClassifier, this project aims to solve the challenge posed by the redaction of sensitive information of names in documents, which is a common practice in industries handling confidential data such as legal documents, medical records, and government files.

# Project Description / Objective
"The Unredactor" aims to develop an efficient machine learning tool to accurately restore redacted names in textual documents, enhancing data access while ensuring privacy compliance. 

Key objectives include:

Automated Unredaction: Implement machine learning to predict redacted names based on textual context.
Advanced Text Processing: Use NLP to analyze text and extract features crucial for unredaction.
Customization and Scalability: Allow customization for various user needs and ensure the system can handle large datasets efficiently.
High Accuracy and Detailed Reporting: Achieve high accuracy in name restoration and provide detailed analytics on the process.

# Theoretical Background
The project employs NLP techniques to process text data by tokenizing sentences, removing stopwords and punctuation, and extracting features that are useful for understanding the context around redactions. Machine learning, particularly the RandomForest algorithm, is used due to its efficacy in handling complex classification tasks with high-dimensional data. The RandomForest algorithm offers robustness and accuracy by combining multiple decision trees to improve predictive performance and prevent overfitting.


# Work Flow 

1. Data Preprocessing
Load Data: Read data from TSV files, handling malformed lines to ensure data integrity.
Preprocess Text: Tokenize text and remove stop words using spaCy, preparing it for feature extraction.
2. Feature Engineering
Extract Features: Develop features like redaction length, context length, word count, and part-of-speech tags to enhance model training.
3. Model Training and Validation
4. Train Model: Train a RandomForestClassifier using features transformed by TfidfVectorizer.
5. Validate Model: Evaluate the model on validation data using metrics such as precision, recall, and F1-score.
6. Generating Predictions
Test Predictions: Apply the trained model to the test data, producing predictions that are saved in submission.tsv.


# Project Pipeline Summary
Initialization: Loads the SpaCy en_core_web_md model for robust natural language processing capabilities.

1. Data Loading:
load_data: Reads TSV files into pandas DataFrames, distinguishing between training/validation and test datasets.

2. Text Preprocessing:
preprocess_text_spacy: Standardizes text by removing stop words and punctuation for cleaner data input.

3. Context Extraction:
extract_full_context: Isolates and reconstructs textual context around redaction markers to preserve essential information.

4. Feature Engineering:
Utilizes TfidfVectorizer to convert texts into a matrix of TF-IDF features, emphasizing important words within documents.

5. Model Optimization:
perform_grid_search: Fine-tunes RandomForestClassifier settings through grid search to enhance model accuracy.

6. Model Evaluation:
evaluate_model: Uses metrics like precision, recall, and F1-score to assess the trained model's performance.

7. Prediction and Output:
Final model predictions are made on the test data and stored in submission.tsv, which maps each text entry’s ID to the predicted name.



# Installation Guide
Install pipenv for virtual environment and package management:

bash

pip install pipenv

# Install project dependencies:

bash

pipenv install tensorflow tensorflow-hub

#Download necessary machine learning models:

bash

pipenv run python -m spacy download en_core_web_sm
pipenv run python -m tensorflow_hub

# Set up the virtual environment: Navigate to your project directory and create a pipenv environment with Python 3.12:

bash
cd the-unredactor
pipenv --python 3.12

# How to Run
Once the installation is complete, you can run the project directly from the command line using the following command:

bash
pipenv run python main.py

This command runs the main.py script, which is the main entry point of "The Unredactor" system. Make sure you are in the project directory where main.py is located when you execute this command.


# Modules and Function Descriptions
main.py integrates various modules and functions to construct a robust pipeline for predicting and unredacting names in text documents. Below are the primary functions and their roles:

1. Load SpaCy for NLP
Functionality: Initializes the spaCy library to use its NLP capabilities, crucial for text preprocessing.
Key Component Used: nlp = spacy.load("en_core_web_sm")

2. load_data(file_path, is_test=False)
Purpose: Loads data from a specified TSV file into a DataFrame, distinguishing between training/validation and test datasets.
Parameters:
file_path: Path to the data file.
is_test: Boolean flag to indicate if the dataset is for testing, affecting the DataFrame structure.
Returns: A pandas DataFrame formatted according to the dataset type.

3. preprocess_text_spacy(text)
Purpose: Processes text by tokenizing and removing stop words and punctuation using spaCy, preparing it for feature extraction.
Parameter:
text: String containing the text to preprocess.
Returns: Preprocessed text as a cleaned, lowercased string.

4. apply_preprocessing(data, column)
Purpose: Applies text preprocessing to a specified column in a DataFrame using the preprocess_text_spacy function.
Parameters:
data: DataFrame containing the text data.
column: The name of the column to preprocess.
Returns: Updated DataFrame with the preprocessed text.

5. extract_full_context(text)
Purpose: Removes redaction markers and extracts the surrounding text context, crucial for understanding the content around redactions.
Parameter:
text: Text containing redaction markers.
Returns: Text with redaction markers removed and context concatenated.

6. apply_full_context_extraction(data, column)
Purpose: Applies the extract_full_context function to a specific column in a DataFrame, ensuring that the full context around redactions is captured.
Parameters:
data: DataFrame containing the data.
column: Column name to apply the extraction to.
Returns: DataFrame with the modified column.

7. perform_grid_search(X_train, y_train)
Purpose: Conducts a grid search to optimize hyperparameters for RandomForestClassifier, aiming to enhance model performance.
Parameters:
X_train: Training dataset features.
y_train: Training dataset labels.
Returns: The RandomForestClassifier instance configured with the best-found parameters.

8. evaluate_model(y_true, y_pred)
Purpose: Evaluates the predictive performance of the model using precision, recall, and F1-score metrics.
Parameters:
y_true: Actual labels.
y_pred: Predicted labels by the model.
Returns: Outputs the performance metrics.

9. main()
Purpose: Orchestrates the complete machine learning pipeline for unredaction, from data loading and preprocessing to model training, evaluation, and prediction.
Workflow:
Loads and preprocesses the data.
Splits data, trains the model, and performs evaluations.
Predicts using the trained model and saves predictions.

# Expected Output
File Creation: Upon completion, "The Unredactor" generates a submission.tsv file.

File Format: The output is a TSV (Tab-Separated Values) file, containing two columns:

ID: Unique identifiers for each text entry from the test dataset.
Name: Names predicted by the model to replace redactions.

# Purpose of the File:

Evaluation: Allows for accuracy assessment against ground truth data.
Record Keeping: Serves as documentation for compliance and auditing.
Further Processing: Can be used for additional data tasks or integrations.


# Bugs and Assumptions
1. Inconsistent Preprocessing: SpaCy's small model might not uniformly standardize text, potentially reducing training effectiveness due to unrecognized entities or formats.
2. Data Imbalance: If training data is not diverse in terms of names and contexts, the model may develop biases, impacting its ability to generalize across different datasets.
3. Performance Metrics Limitations: Relying solely on precision, recall, and F1-score might not adequately reflect the model's effectiveness in real-world applications, particularly when balancing the impact of false positives and negatives.
4. Redaction Marker Consistency: Assumptions that the redaction marker '█' is used uniformly could lead to errors if variations exist, affecting context extraction accuracy.
5. Computational Demands of Grid Search: Grid search for hyperparameter tuning is resource-intensive and may not include the optimal parameter ranges, potentially leading to suboptimal model performance.
6. File Handling and Format Assumptions: The system assumes correct file paths and formats without robust error handling, which could result in runtime failures if discrepancies occur.
7. Dependency on External Libraries: Changes or updates in external libraries like pandas, scikit-learn, or spaCy could impact project stability and performance.
8. Processing Efficiency: The project does not utilize multi-threading or GPU acceleration, which might limit processing speed, especially with larger datasets.












































