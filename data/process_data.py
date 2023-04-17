import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Loads messages and categories from csv file to dataframe , prints information and returns dataframe.
    Input:
        messages_filepath: Filepath to messages csv file
        categories_filepath: Filepath to categories csv file
    Returns:
        df: Combined messages and categories dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    print(50*"*")
    print(f"Loaded Data from {messages_filepath} and {categories_filepath}")
    print("Head of data looks like the following:")
    print(f"{messages_filepath}:\n")
    print(messages.head())
    print(f"{categories_filepath}:\n")
    print(categories.head())
    print(50*'=')
    print("Merging datasets")
    df = messages.merge(categories,on='id', how='left')
    print(df.head())
    return df

def clean_data(df):
    '''
    clean_data
    Cleans dataframe and prints information for later classification
    Input:
        df: Dataframe
    Return: 
        df: Cleaned dataframe
    '''
    print(50*"*")
    print("Cleaning data...")

    categories = df.categories.str.split(";",expand=True)
    row =  categories.iloc[[0]]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.str.split("-")[0][0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # remove related=2 rows
    categories = categories[(categories.related ==1) | (categories.related ==0 )]
    # remove original columns since it contains NaNs
    df = df.drop(columns=["original"], axis=1)
    # drop the original categories column from `df`
    df = df.drop(columns=["categories"], axis=1)
    df = pd.concat([df, categories], axis=1)
    #remove duplicates
    df = df.drop_duplicates()

    print(50*"=")
    print("Cleaned up data:")
    print(df.head())

    return df

def save_data(df, database_filename):
    '''
    save_data
    Saves data in an sqlite DB which can used by a classification model subsequently.
    Input:
        df: Dataframe which should be stored in the DB.
        database_filename: Filename of DB.
    '''
    print(50*"*")
    print(f"Saving data to database {database_filename} or replacing it...")
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('PolishedDisasterData', engine, index=False, if_exists="replace")
    print("Successfully saved!")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        #print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
        #      .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        #print('Cleaning data...')
        df = clean_data(df)
        
        #print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        #print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()