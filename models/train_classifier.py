import sys
# import libraries
import sqlite3
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
#ML Pipelines
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    '''
    input:
        database_filepath: Input of Database.db file for SQLite. 
    output:
        df: dataframe. dataframe reading the SQLite db
        X: dataframe. Dataframe with Message column
        y: dataframe. Dataframe that have dropped several rows from original df. 
    '''
    engine = create_engine('sqlite:///disaster.db')
    df = pd.read_sql_table('disaster', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = list(y.columns.values)
    return X, y, category_names

def tokenize(text):
    """
    Normalize, tokenize and stem text into words
    
    Args:
    text, a string of words 
       
    Returns:
    stem, array of strings containing words
    """
    #lower case and remove special punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ",text.lower())
    #split using tokenizer
    words = word_tokenize(text)
    #remove stopwords to reduce vocab & use stem
    words = [w for w in words if w not in stopwords.words("english")]
    
    return words


def build_model():
    """
    Build our ML pipeline model 
    Args: Pipeline, Training data
    Returns: Dataframe. GridsearchCV data
    
    """
    pipeline_dtc = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(DecisionTreeClassifier())),
    ])
    
    dtc_params = {
        'tfidf__smooth_idf':[False]
    }

    cv = GridSearchCV(pipeline_dtc, dtc_params, cv=3, n_jobs=-1)
    return cv

def pred_loop(actual, predicted, col_names):
    
    """
    Args:
    actual: Array with labels
    predicted: Array with labels
    col_names: Names for each column
       
    Returns:
    predictions_df: Dataframe with recall, precision, f1 and accuracy scores
    """
    metrics = []
    
    #Loop to score each of the metrics and predicitions for inputted arrays
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i], average='micro')
        recall = recall_score(actual[:, i], predicted[:, i], average='micro')
        f1 = f1_score(actual[:, i], predicted[:, i], average='micro')
        
        metrics.append([accuracy, precision, recall, f1])
    
    #Dataframe creation containing the predictions
    metrics = np.array(metrics)
    predictions_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return predictions_df


def evaluate_model(model, X_test, y_test, category_names):
    """
    Returns accuracy, precision, recall and F1 scores of the data
    Args: 
    model: Fitted model
    X_test: Dataframe containing test features dataset.
    Y_test: Dataframe containing test labels dataset.
    category_names: List containing category names.
    """
    y_pred = model.predict(X_test)
    model_loop = pred_loop(np.array(y_test), y_pred, category_names) 
    print(model_loop)

def save_model(model, model_filepath):
    """
    Save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    