# app.py
from flask import Flask, request, jsonify
import sqlite3
import os
import pandas as pd
import re
from pysentimiento import create_analyzer
from datetime import datetime, timezone, timedelta
from collections import defaultdict

app = Flask(__name__)

# Function to calculate sentiment
def analyze_sentiment(message):
    analyzer = create_analyzer(task="sentiment", lang="es")
    return analyzer.predict(message)  

# Function to extract messages from the chat.db SQLite file
def get_messages_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    all_messages = pd.read_sql_query("select * from message", conn)
    all_handles = pd.read_sql_query("select * from handle", conn)

    all_messages.rename(columns={'ROWID' : 'message_id'}, inplace = True)
    all_handles.rename(columns={'id' : 'phone_number', 'ROWID': 'handle_id'}, inplace = True)
    merged = pd.merge(all_messages[['text', 'handle_id', 'date','is_from_me', 'message_id']], all_handles[['handle_id', 'phone_number']], on ='handle_id', how='left')
    
    return merged

def gather_messages(phone_number):
    messages_df = merged.loc[merged['phone_number'] == phone_number]
    
    my_messages_df = messages_df[messages_df['is_from_me'] == 1]
    my_messages_list = list(zip(my_messages_df.text, my_messages_df.date))
    
    other_messages_df = messages_df[messages_df['phone_number'] == 0]
    other_messages_list = list(zip(other_messages_df.text, other_messages_df.date))
    
    return {'my_messages_df': my_messages_df,
            'my_messages_list': my_messages_list,
            'other_messages_df': other_messages_df,
            'other_messages_list': other_messages_list
           }

emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F" 
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F700-\U0001F77F"  
        "\U0001F780-\U0001F7FF"  
        "\U0001F800-\U0001F8FF" 
        "\U0001F900-\U0001F9FF" 
        "\U0001FA00-\U0001FA6F" 
        "\U0001FA70-\U0001FAFF"  
        "\U00002700-\U000027BF"  
        "\U0001F1E6-\U0001F1FF"
        "\U00002300-\U000023FF"
                           "]+", flags=re.UNICODE)

url_pattern = re.compile("https")

touch_pattern = re.compile(r'\bDigital Touch Message\b')

misc_unicode_pattern = re.compile(r"\\U")

def _clean_message(message):
    cleaned = _clean_empty(message)
    cleaned = _clean_emojis(cleaned)
    cleaned = _clean_urls(cleaned)
    cleaned = _clean_touch_message(cleaned)
    cleaned = _clean_misc(cleaned)
    return cleaned

def _clean_empty(message):
    return "" if message is None else message

def _clean_emojis(message):
    return emoji_pattern.sub(r'', message)

def _clean_urls(message):
    return "" if url_pattern.search(message) else message

def _clean_touch_message(message):
    return "" if touch_pattern.search(message) else message

def _clean_misc(message):
    return "" if misc_unicode_pattern.search(message) else message

def calibrate_timestamp(timestamp):
    scaled_timestamp = timestamp / 1e9
    epoch_difference = 978307200
    unix_timestamp = scaled_timestamp + epoch_difference
    date = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc) - timedelta(hours=4)
    return date

def clean_messages(messages):
    cleaned_list = []
    for message, date in messages:
        cleaned_list.append((_clean_message(message), calibrate_timestamp(date)))
    return cleaned_list

# Route to handle file upload and sentiment analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # save the uploaded file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract messages and perform sentiment analysis
    messages = get_messages_from_db(file_path)
    sentiment_results = [{"message": msg, "sentiment": analyze_sentiment(msg)} for msg in messages]

    return jsonify({"sentiment_results": sentiment_results})

if __name__ == '__main__':
    app.run(debug=True)
