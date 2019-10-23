from waitress import serve
from flask import Flask, render_template, request, url_for, make_response
from time import strftime
import time
import datetime
import regex
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import requests
from io import BytesIO
import imblearn
from imblearn.pipeline import Pipeline
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("agg")

app = Flask(__name__, static_url_path="/static")

@app.route("/")
def index():
    """Return the main page."""
    time_str = strftime("%m/%d/%Y %H:%M")
    print(time_str)
    return render_template("index.html", time_info=time_str)


# Functions necessary for the website model to run

def get_clean_text_pattern(recomposed_note):
    """Function that filters through the notes, retrieves those that match
     the specified pattern and removes stopwords."""
    pattern = "([a-zA-Z0-9\\\]+(?:'[a-z]+)?)"
    recomposed_note_raw = nltk.regexp_tokenize(recomposed_note, pattern)
    # Create a list of stopwords and remove them from our corpus
    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    # additional slang and informal versions of the original words had to be added to the corpus.
    stopwords_list += (["im", "ur", "u", "'s", "n", "z", "n't", "brewskies", "mcd’s", "Ty$",
                        "Diploooooo", "thx", "Clothessss", "K2", "B", "Comida", "yo", "jobby",
                        "F", "jus", "bc", "queso", "fil", "Lol", "EZ", "RF", "기프트카드", "감사합니다",
                        "Bts", "youuuu", "X’s", "bday", "WF", "Fooooood", "Yeeeeehaw", "temp",
                        "af", "Chipoodle", "Hhuhhyhy", "Yummmmers", "MGE", "O", "Coook", "wahoooo",
                        "Cuz", "y", "Cutz", "Lax", "LisBnB", "vamanos", "vroom", "Para", "el", "8==",
                        "bitchhh", "¯\\_(ツ)_/¯", "Ily", "CURRYYYYYYY", "Depósito", "Yup", "Shhhhh"])

    recomposed_note_stopped = ([w.lower() for w in recomposed_note_raw if w not in stopwords_list])
    return recomposed_note_stopped


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_notes(recomposed_note_stopped):
    "Function that lemmatizes the different notes."
    # Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_notes = []
    for sentence in recomposed_note_stopped:
        for word in nltk.word_tokenize(sentence):
            lem = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            lemmatized_notes.append(lem)
    return lemmatized_notes


def get_user_details_df():
    "Function that from the request form returns a dataframe with all the inputs for our model."
    data = request.form
    user_details = {}
    amount = 0
    exchange = []
    for response in data:
        if response == 'amount':
            amount += int(data[f'{response}'])
        elif response == 'local_currency':
            exchange.append(data[f'{response}'])
        elif response == 'desired_currency':
            exchange.append(data[f'{response}'])
        elif response == 'time_since_account_inception':
            user_details[response] = (datetime.datetime.today() - datetime.datetime.strptime(data[f'{response}'], '%d-%m-%Y')).total_seconds()
        elif response == 'max_time_between_transactions':
            user_details[response] = int(data[f'{response}'])*3600  # turning hours to seconds
        elif response == 'mean_time_between_transactions':
            user_details[response] = int(data[f'{response}'])*3600  # turning hours to seconds
        elif response == 'text_description':
            # Dealing with the text aspect
            recomposed_note_stopped = get_clean_text_pattern(data[f'{response}'])
            lemmatized_notes = lemmatize_notes(recomposed_note_stopped)
            # Load the vectorizer model
            vectorizer = Doc2Vec.load("d2v.model")
            # Find the vectors for each note in the whole note corpus
            _vectrs = np.array(vectorizer.infer_vector(lemmatized_notes))
            for i in range(len(_vectrs)):
                user_details[f'text_vector_{i}'] = _vectrs[i]
        else:
            user_details[response] = int(data[f'{response}'])
    user_details['n_transactions_made_during_week'] = (
        user_details['n_transactions_made'] -
        user_details['n_transactions_made_during_weekend']
    )
    # transform the user_details dictionary to a dataframe
    user_details_df = pd.DataFrame([user_details])
    return user_details_df, amount, exchange


def get_keys(path):
    with open(path) as f:
        return json.load(f)


def get_fx_rates(exchange):
    """Function that returns the 100 day FX rate history for the currency wished to
    be exchanged in json format."""
    url = 'https://www.alphavantage.co/query?'
    function_input = 'FX_DAILY'
    # get api key
    keys = get_keys(".secret/alphaadvantage.json")
    api_key = keys['api_key']
    # get user details
    # user_details, exchange = get_user_details_df()
    from_symbol_input = exchange[0]
    to_symbol_input = exchange[1]
    url_params = (f"""function={function_input}&from_symbol={from_symbol_input}&to_symbol={to_symbol_input}&apikey={api_key}""")
    request_url = url + url_params
    response = requests.get(request_url)
    return response


def get_adjusted_rate(response_json):
    "Function that converts json into pd dataframe with historic adj closed prices."
    response_dict = {}
    for key, val in response_json.json()['Time Series FX (Daily)'].items():
        response_dict[key] = float(val['4. close'])
    response_df = pd.DataFrame.from_dict(response_dict, 'index')
    response_df.columns = ['Adj Close Price']
    response_df = response_df.reindex(index=response_df.index[::-1])
    return response_df


def get_bollinger_bands(response_df):
    """Function that returns the bollinger bands for the exchange rate in question."""
    response_df['30 Day MA'] = response_df['Adj Close Price'].rolling(window=20).mean()
    response_df['30 Day STD'] = response_df['Adj Close Price'].rolling(window=20).std()
    response_df['Upper Band'] = response_df['30 Day MA'] + (response_df['30 Day STD'] * 2)
    response_df['Lower Band'] = response_df['30 Day MA'] - (response_df['30 Day STD'] * 2)
    return response_df


def get_graphical_view(response_df, exchange_currency, desired_currency, today):
    """Function that returns a graphic view of the exchange rate in question
    and the corresponding bollinger bands."""
    # We only want to show the previous month, therefore subset the dataframe
    one_month_ago = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=today.day).strftime("%Y-%m-%d")
    date_15_days_ago = (today - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    response_df = response_df.loc[(response_df.index >= one_month_ago) & (response_df.index <= today.strftime("%Y-%m-%d"))]
    
    # set style, empty figure and axes
    fig = plt.figure(figsize=(10,5), facecolor='w')
    ax = fig.add_subplot(111)
    
    # Get index values for the X axis for exchange rate DataFrame
    x_axis = response_df.index
    
    # Plot shaded 21 Day Bollinger Band for exchange rate
    #ax.fill_between(x_axis, response_df['Upper Band'], response_df['Lower Band'], color='white')
    
    # Plot Adjust Closing Price and Moving Averages
    ax.plot(x_axis, response_df['Adj Close Price'], color='blue', lw=2)
    #ax.plot(x_axis, response_df['30 Day MA'], color='black', lw=2)
    ax.plot(x_axis, response_df['Upper Band'], color='green', lw=2, )
    ax.plot(x_axis, response_df['Lower Band'], color='red', lw=2)
    ax.set_xticks([one_month_ago, date_15_days_ago, today.strftime("%Y-%m-%d")])
    ax.yaxis.tick_right()
    ax.set_facecolor('#ffffff')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Set Title & Show the Image
    # Compare the value of the exchange rate currencies
    compare = response_df.loc[response_df.index == today.strftime("%Y-%m-%d")]
    if compare['Adj Close Price'].values > compare['Upper Band'].values:
        print(f'The {exchange_currency} is strong, consider making your international transaction today.')
    elif compare['Adj Close Price'].values > compare['Lower Band'].values:
        print(f"The {exchange_currency} is currently trading according to its boundaries.")
    else:
        print(f"The {exchange_currency} is weak, consider making your international transaction another day.")
    plt.savefig("static/exchange_rate_graph.png")
    fileobject = BytesIO()
    plt.savefig(fileobject, format="png")
    response = make_response(fileobject.getvalue())
    response.headers["Content-Type"] = 'image/png'
    return response


@app.route("/get_exchange_rate_graph", methods=["GET"])
def get_exchange_rate_graph():
    """Function that takes in the user's prefered exchange rate change and 
    outputs a graph with the corresponding bollinger bands"""
    # get inputs
    # user_details, exchange = get_user_details_df()
    local_currency = request.args.get('local_currency', '')
    desired_currency = request.args.get('desired_currency', '')
    response_json = get_fx_rates([local_currency, desired_currency])
    response_df = get_adjusted_rate(response_json)
    response_bb_df = get_bollinger_bands(response_df)
    today = datetime.date.today()
    response_graph = get_graphical_view(response_bb_df, local_currency,
                                        desired_currency, today)
    return response_graph


@app.route("/get_results", methods=["POST"])
def get_results():
    """Function that predicts whether a user is going to make a transaction in
    the next two days or not based on their form input."""
    # get inputs
    user_details, amount, exchange = get_user_details_df()
    # load the model from disk
    red_ent_forest_model = joblib.load('red_forest_pipe.joblib')
    # prediction = red_ent_forest_model.predict_proba(user_details_ss)
    prediction = red_ent_forest_model.predict_proba(user_details)
    response_json = get_fx_rates(exchange)
    response_df = get_adjusted_rate(response_json)
    response_bb_df = get_bollinger_bands(response_df)
    close_price = response_bb_df[-1:]['Adj Close Price'][0]
    mean_price = response_bb_df[-1:]['30 Day MA'][0]
    upper_band = response_bb_df[-1:]['Upper Band'][0]
    lower_band = response_bb_df[-1:]['Lower Band'][0]
    print(type(close_price))
    return render_template("results.html", prediction=prediction,
                           amount=amount, exchange=exchange,
                           close_price=close_price, upper_band=upper_band,
                           lower_band=lower_band, mean_price=mean_price)

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000, threads=1)