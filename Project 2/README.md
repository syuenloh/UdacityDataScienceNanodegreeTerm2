# Disaster Response Pipeline Project

### Files:
This folder contains all the files for the Disaster Response Pipeline Project, which outputs a web app. There are 3 folders:
- app - contains files to render the web app. 
- data - contains ETL pipeline files to clean and save data
- models - contains file to train and save model

### Instructions:
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
