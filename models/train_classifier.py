import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def load_data(database_filepath):
    '''
    load_data
    Loads data from an sqlite DB and splits it into X,Y dataarrays.
    Input:
        database_filepath: Filepath of DB.
    Output:
        X: Feature array.
        Y: Multidimensional target array.
        column_names: Target column-names.
    '''
    print(50*"*")
    print(f"Create connection to DB {database_filepath}")
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table('PolishedDisasterData', conn)
    X = df.message
    Y = df[df.columns[4:]]
    print(50*"=")
    print("Preview of X and Y\n")
    print("X:\n")
    print(X.head())
    print("\nY:\n")
    print(Y.head())

    # Search for NaNs again since DB read might be messy
    Y_nan = Y[Y.isna().any(axis=1)]
    Y = Y.drop(Y_nan.index)
    X = X.drop(Y_nan.index)
    X = X.values
    Y = Y.values
    column_names = df.columns[4:]

    return X,Y, column_names

def tokenize(text):
    import nltk
    nltk.download(['punkt', 'wordnet'], quiet=True) 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model
    Sets up data pipeline and gridsearch via cross validation for classifying disaster messages.
    '''
    pipeline = Pipeline([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
            'clf__estimator__n_estimators': [10],
            #'clf__estimator__min_samples_split': [2]
        }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=6, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    Evaluates model by predicting targets and prints statistic obervables.
    Input:
        model: Classification model
        X_test: Test features
        Y_test: Test targets. Multidimensional array
        category_names: Names of classification categories.

    '''
    Y_pred = model.predict(X_test)
    print(Y_test.shape)
    print(Y_pred.shape)
    for col_num in range(Y_test.shape[1]-1):
        category_name = category_names[col_num]
        print(category_name)
        print(50*"=")
        print("column: " + str(col_num))
        print(classification_report(Y_test[:,col_num], Y_pred[:,col_num], target_names=[category_name, 'no ' + category_name], labels=[0,1]))

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test,Y_pred, average=None)
    recall = recall_score(Y_test, Y_pred, average=None)
    print("\nBest Parameters:", model.best_params_)
    print("Accuracy:", accuracy.mean())
    print("Precision:", precision.mean())
    print("Recall: ", recall.mean())
    print("F1-score: ", 2*(recall.mean()*accuracy.mean())/(recall.mean()+accuracy.mean()))

def save_model(model, model_filepath):
    '''
    save_model
    Saves model to pickle file
    Input:
        model: Classification model
        model_filepath: Model-Filepath
    '''
    with open(model_filepath, 'wb') as files:
        pickle.dump(model, files)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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