import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Returns merged dataframe from messages and categories datasets on the common id.
    Inputs:
    - messages_filepath: filepath to messages dataset
    - categories_filepath: filepath to categories dataset
    """ 
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    df=pd.merge(messages,categories,on='id')
    return df

def clean_data(df):
    """Returns cleaned dataframe where:
        - Values in the categories column are split so that each value becomes a separate column.
        - Columns of categories are renamed new column names.
        - String values encoded into numeric values of 0 or 1.
        - Duplicates dropped.
        
        Input: dataframe to be cleaned
        """
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[:][-1])
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # replace all values that equals 2 to be encoded as 1
    categories= categories.replace(2, 1)    
    
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saves the dataframe into an sqlite database.
    Inputs:
    - df: dataframe to be saved
    - database_filename: name of database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists='replace',chunksize=500)  


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
