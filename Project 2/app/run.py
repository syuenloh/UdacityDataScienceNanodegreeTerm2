import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import defaultdict


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    def message_count(val):
        return dict((cat,sum([ val== x for x in list(df[cat].values)])) for cat in df.columns[-36:])

    zero_list=message_count(0)
    one_list=message_count(1)
    two_list=message_count(2)

    dd = defaultdict(list)

    for d in (zero_list,one_list,two_list): # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)

    df_new=pd.DataFrame(dd).transpose()
    categories=list(df_new.index)
    message_count=df_new[1]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        dict(
            data=[
                Bar(
                x=categories,
                y=message_count,
                marker=dict(
            color='rgba(189,31,201,1)')
                ),
            ],
            layout=dict(
                title='Distribution of messages by categories classified as 1'
            ) 
        ),
        dict(
            data=[
                Bar(
                    x=categories,
                    y=df_new[0],
                 marker=dict(
            color='rgba(58,218,250,1)')),
            ],
            layout=dict(
                title='Distribution of messages by categories classified as 0'
            ) 
        ),
        dict(
            data=[
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            
            layout=dict(
                title='Distribution of Message Genres',
                yaxis= dict(title="Count"),
                xaxis=dict(title="Genre")
            ) 
          
        )
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
