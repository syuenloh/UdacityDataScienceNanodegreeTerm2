# Disaster Response Pipeline Project

### Files:
This folder contains all the files for the Disaster Response Pipeline Project, which outputs a web app. There are 3 folders:
- data - contains ETL pipeline files to clean and save data
- models - contains file to train and save model
- app - contains files to render the web app. 

#### `data` folder
The data folder contains the file `process.py` which contains code to load, clean and save the data in an SQLite database. 

It contains a function to load the data from provided filepaths for messages and categories datasets, a cleaning function to clean and merge the datasets, and saving the data into a database. 

When running the file, you will need to provide the filepaths of the messages and categories datasets as the first and second argument respectively, and the filepath of the database to save the cleaned data to as the third argument.

#### `models` folder
The models folder contains the file `train_classifier.py` which contains code to load the data, clean the text data, build, train and optimize a simple natural language processing classification model, and export the trained model. 

The code contains a load data function that loads the previously cleaned data from the data folder. 
The tokenize function will tokenize and clean the text data to extract features. 
The build model function in the code will build a model via a pipeline and use grid search to find the best hyperparameters for the model, while the evaluate model function will print the classification report for each output category of the model.
Finally the model will be saved and exported as a pickle file using the `save_model` function.

#### `app` folder
The app folder contains the folder `templates`, and the file `run.py`. The templates file contains the html files needed to render the web app. Whereas the file `run.py` extracts the data needed for visuals and renders the charts on the web app, and outputs the model output as a dynamic chart. 

### Instructions:
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
