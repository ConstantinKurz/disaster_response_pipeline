# Disaster Response Pipeline Project
### Introduction
Disaster Response Pipeline uses Machine Learning to predict a class for incoming messenges using CountVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) and TfidfTransformer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
Models are trained using a MultiOutput RandomForest(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
Prediction can then help to fast classify incoming messenges in case of a given disaster or to help identify a disaster.

### Project structure
- app
    - template
        - master.html # main page of web app
        - go.html # classification result page of web app
        - run.py # Flask file that runs app
- data
    - disaster_categories.csv # data to process
    - disaster_messages.csv # data to process
    - process_data.py
    - InsertDatabaseName.db # database to save clean data to
- models
    - train_classifier.py
    - classifier.pkl # saved model not present.
- README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
