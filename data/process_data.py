import sys
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
    Load then merge the messages and categories datasets
    
    Args:
    messages_filepath:path to csv file containing messages dataset.
    categories_filepath:path to csv file containing categories dataset.
       
    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on='id')

    return df


def clean_data(df):
    """
    Cleans the dataframe by slicing and rmeoving duplicated which are replaced as binaries
    
    Args:
    df: dataframe. Dataframe containing messages and categories in a merged dataframe
       
    Returns:
    df: dataframe. cleaned dataframe. 
    """
    
   # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.transform(lambda x: x[0:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(str)
    
    # drop the original categories column from `df`
    df=df.drop(['categories'], axis=1) 
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Saves the cleaned datafram into a SQLite3 DB. 
    
    Args:
    df: dataframe. Dataframe containing cleaned version of merged message and 
    categories data.
    database_filename: string. Filename for output database.
       
    Returns: DB File in outpath
    
    """  
    engine = create_engine('sqlite:///disaster.db')
    df.to_sql('disaster', engine, index=False)
    
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()