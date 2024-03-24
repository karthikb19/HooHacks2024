from flask import Flask, redirect, request, session, url_for, render_template
import requests
from urllib.parse import urlencode
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timezone, timedelta
import secrets
from tqdm import tqdm
from joblib import dump, load
from sklearn.utils.class_weight import compute_class_weight
import pytz
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


# Constants for Dexcom API OAuth 2.0 authentication
CLIENT_ID = 'Xv8e7QwMcm3jBHztPipV6tMEP6QFH4Zt'
CLIENT_SECRET = 'rYww4k3RwxdYWPFo'
REDIRECT_URI = 'http://127.0.0.1:5001/callback'
AUTHORIZE_URL = 'https://sandbox-api.dexcom.com/v2/oauth2/login'
TOKEN_URL = 'https://sandbox-api.dexcom.com/v2/oauth2/token'
SCOPE = 'offline_access'

app = Flask(__name__)
app.secret_key = 'juk'  # Change this to a random secret key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    # Check if the user is logged in by checking the session
    if 'access_token' not in session:
        return redirect(url_for('login'))
    
    if 'model_trained' not in session:
        return redirect(url_for('train'))
    
    headers = check_refresh()
    
    range_url = "https://sandbox-api.dexcom.com/v3/users/self/dataRange"
    range_data = fetch_data(range_url, {}, headers)
    end_date = safe_strptime(range_data['events']['end']['systemTime'])
    start_date = end_date - timedelta(minutes=31)

    egvs_url = "https://sandbox-api.dexcom.com/v3/users/self/egvs"
    egvs_query = {
        "startDate": start_date.strftime('%Y-%m-%dT%H:%M:%S'),
        "endDate": end_date.strftime('%Y-%m-%dT%H:%M:%S')
    }
    egvs_data = fetch_data(egvs_url, egvs_query, headers)
    x_values = [egv['value'] for egv in egvs_data.get('records', []) if egv['value'] is not None][:6][::-1]
    graphx_values = [egv['systemTime'] for egv in egvs_data.get('records', []) if egv['value'] is not None][:6][::-1]
    scaler = load('scaler.joblib')
    x_values_normalized = scaler.transform(np.array(x_values).reshape(1, -1))
    x_values_reshaped = np.reshape(x_values_normalized, (1, 6, 1))

    model = load_model('lstm_model.h5')
    y_pred = model.predict(x_values_reshaped)
    pred = int(round(y_pred[0][0]))
    print(str(y_pred[0][0]))
    utc_zone = pytz.utc
    est_zone = pytz.timezone('US/Eastern')
    x_est = [utc_zone.localize(safe_strptime(date)).astimezone(est_zone).strftime('%H:%M') for date in graphx_values]

    if pred == 1:
        scaler = load('scaler_insulin.joblib')
        model = load_model('lstm_insulin_model.h5')
        x_values_normalized = scaler.transform(np.array(x_values).reshape(1, -1))
        x_values_reshaped = x_values_normalized.reshape((1, 6, 1))
        y_pred = model.predict(x_values_reshaped)
        insulin_dose = round(float(y_pred[0][0]), 1)
        meal = f"A meal was detected in the past 30 minutes. Our algorithm recommends an insulin dose of ~{insulin_dose} units."

    else:
        meal = "No meal was detected in the past 30 minutes. Continue monitoring blood glucose levels."

    return render_template('user.html', x_values=x_est, y_values=x_values, meal=meal)

@app.route('/train')
def train():
    # Check if the user is logged in by checking the session
    if 'access_token' not in session:
        return redirect(url_for('login'))
    
    return redirect(url_for('predict'))

    headers = check_refresh()
    
    range_url = "https://sandbox-api.dexcom.com/v3/users/self/dataRange"
    range_data = fetch_data(range_url, {}, headers)

    # Fetch Data
    # end_date = safe_strptime(range_data['events']['end']['systemTime'])
    # start_date = end_date - timedelta(days=30)
    # egvs_df, events_df = fetch_and_process_data(start_date, end_date, headers)
    # egvs_df.to_csv('egvs_data.csv', index=False)
    # events_df.to_csv('events_data.csv', index=False)

    egvs_df, events_df = pd.read_csv('egvs_data.csv'), pd.read_csv('events_data.csv')
    ml(egvs_df, events_df)

    # Fetch Insulin Data
    # end_date = safe_strptime(range_data['events']['end']['systemTime'])
    # start_date = end_date - timedelta(days=90)
    # insulin_egvs_df, insulin_events_df = fetch_and_process_insulin_data(start_date, end_date, headers)
    # insulin_egvs_df.to_csv('insulin_egvs_data.csv', index=False)
    # insulin_events_df.to_csv('insulin_events_data.csv', index=False)

    insulin_egvs_df, insulin_events_df = pd.read_csv('insulin_egvs_data.csv'), pd.read_csv('insulin_events_data.csv')
    insulin_ml(insulin_egvs_df, insulin_events_df)
    
    session["model_trained"] = True
    return redirect(url_for('predict'))

def ml(egvs_df, events_df):
    # Find indices with NaN values in each dataframe
    nan_indices_egvs = set(egvs_df[egvs_df.isna().any(axis=1)].index)
    nan_indices_events = set(events_df[events_df.isna().any(axis=1)].index)

    # Combine indices from both dataframes
    combined_nan_indices = nan_indices_egvs.union(nan_indices_events)

    # Drop the combined indices from both dataframes
    egvs_df = egvs_df.drop(index=combined_nan_indices)
    events_df = events_df.drop(index=combined_nan_indices)

    # Normalize the x values for better neural network performance
    scaler = MinMaxScaler()
    x_values = scaler.fit_transform(egvs_df)

    # For binary classification, we don't need to one-hot encode the y values
    y_values = events_df['eventType'].values

    # Reshape the input data to be 3D [samples, timesteps, features] as required by LSTM
    x_reshaped = np.reshape(x_values, (x_values.shape[0], x_values.shape[1], 1))

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y_values, test_size=0.2)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.flatten())
    class_weight_dict = dict(enumerate(class_weights))

    # Use class weights in model training
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test), class_weight=class_weight_dict, callbacks=[early_stopping])

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # Save the model and the scaler
    model.save('lstm_model.h5')
    dump(scaler, 'scaler.joblib')

def fetch_and_process_data(start_date, end_date, headers):
    # Prepare to store processed data
    x_data = []
    y_data = []

    # Calculate the total number of days to process
    total_days = (end_date - start_date).days

    # Process data in 30-day intervals
    for offset in tqdm(range(0, total_days, 30)):
        interval_start = start_date + timedelta(days=offset)
        interval_end = min(end_date, interval_start + timedelta(days=30))

        # Fetch all EGVs data for the current 30-day interval
        egvs_url = "https://sandbox-api.dexcom.com/v3/users/self/egvs"
        egvs_query = {
            "startDate": interval_start.strftime('%Y-%m-%dT%H:%M:%S'),
            "endDate": interval_end.strftime('%Y-%m-%dT%H:%M:%S')
        }
        all_egvs_data = fetch_data(egvs_url, egvs_query, headers)

        # Fetch all event data for the current 30-day interval
        events_url = "https://sandbox-api.dexcom.com/v3/users/self/events"
        events_data = fetch_data(events_url, egvs_query, headers)  # Reuse egvs_query for dates

        # Filter events based on status
        valid_events = [event for event in events_data.get('records', []) if event.get('eventStatus') == 'created']

        # Initialize the start of the first 31-minute interval
        current_interval_start = interval_start

        # Iterate through 31-minute intervals within the 30-day period
        while current_interval_start + timedelta(minutes=31) <= interval_end:
            current_interval_end = current_interval_start + timedelta(minutes=31)

            # Check if the interval overlaps with any valid event
            overlapping_event = next((event for event in valid_events if current_interval_start <= safe_strptime(event['systemTime']) < current_interval_end), None)

            # Filter EGVs data for this interval
            skip = True
            if overlapping_event:
                current_interval_start = safe_strptime(overlapping_event['systemTime'])
                current_interval_end = min(current_interval_start + timedelta(minutes=31), interval_end)
                # Interval overlaps with a valid event
                if overlapping_event['eventType'] == 'carbs':
                    y_values = {'eventType': 1}
                else:
                    skip = False # Skip this interval if the event is not of interest
            else:
                # Interval does not overlap with any event, considered as control group
                y_values = {'eventType': 0}
            
            interval_egvs = [egv for egv in all_egvs_data.get('records', []) if current_interval_start <= safe_strptime(egv['systemTime']) < current_interval_end]
            x_values = [egv['value'] for egv in interval_egvs if (egv['value'] is not None) and skip][::-1]

            if x_values:  # Only include intervals with EGV data
                while len(x_values) < 6:
                    x_values.append(np.nan)
                x_data.append(x_values[:6])
                y_data.append(y_values)
            
            while len(x_values) < 6:
                    x_values.append(np.nan)
            # Move to the next interval
            current_interval_start = current_interval_end

    # Convert the lists to pandas DataFrames
    x_df = pd.DataFrame(x_data)
    y_df = pd.DataFrame(y_data)

    return x_df, y_df

def fetch_and_process_insulin_data(start_date, end_date, headers):
    # Prepare to store processed data
    x_data = []
    y_data = []

    # Calculate the total number of days to process
    total_days = (end_date - start_date).days

    # Process data in 30-day intervals
    for offset in tqdm(range(0, total_days, 30)):
        interval_start = start_date + timedelta(days=offset)
        interval_end = min(end_date, interval_start + timedelta(days=30))

        # Fetch all EGVs data for the current 30-day interval
        egvs_url = "https://sandbox-api.dexcom.com/v3/users/self/egvs"
        egvs_query = {
            "startDate": interval_start.strftime('%Y-%m-%dT%H:%M:%S'),
            "endDate": interval_end.strftime('%Y-%m-%dT%H:%M:%S')
        }
        all_egvs_data = fetch_data(egvs_url, egvs_query, headers)

        # Fetch all event data for the current 30-day interval
        events_url = "https://sandbox-api.dexcom.com/v3/users/self/events"
        events_data = fetch_data(events_url, egvs_query, headers)  # Reuse egvs_query for dates

        # Filter events based on status
        valid_events = [event for event in events_data.get('records', []) if event.get('eventStatus') == 'created' and event.get('eventType') == 'insulin' and event.get('eventSubType') == 'fastActing']

        for event in valid_events:
            event_time = safe_strptime(event['systemTime'])
            start_time = event_time - timedelta(minutes=31)
            
            # Filter EGVs data for the 31-minute interval before the event
            interval_egvs = [egv for egv in all_egvs_data.get('records', []) if start_time <= safe_strptime(egv['systemTime']) < event_time]
            if interval_egvs:
                y_value = event['value']
                x_values = [egv['value'] for egv in interval_egvs if egv['value'] is not None][::-1]
                
                while len(x_values) < 6:
                    x_values.append(np.nan)
                
                x_data.append(x_values[:6])  # Ensure only the first 6 values are used
                y_data.append(y_value)
                print(x_values[:6], y_value)

    # Convert the lists to pandas DataFrames
    x_df = pd.DataFrame(x_data)
    y_df = pd.DataFrame(y_data, columns=['value'])

    return x_df, y_df

def insulin_ml(egvs_df, events_df):
    events_df['value'] = pd.to_numeric(events_df['value'], errors='coerce')
    # Find indices with NaN values in each dataframe
    nan_indices_egvs = set(egvs_df[egvs_df.isna().any(axis=1)].index)
    nan_indices_events = set(events_df[events_df.isna().any(axis=1)].index)

    # Combine indices from both dataframes
    combined_nan_indices = nan_indices_egvs.union(nan_indices_events)

    # Drop the combined indices from both dataframes
    egvs_df = egvs_df.drop(index=combined_nan_indices)
    events_df = events_df.drop(index=combined_nan_indices)

    # Normalize the x values for better neural network performance
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(egvs_df)

    # Reshape the input to be 3D [samples, timesteps, features] for LSTM
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X_reshaped, events_df["value"].values, test_size=0.2)

    # Define the LSTM model for regression
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for regression

    # Compile the model for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Use early stopping in model training
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=200, validation_split=0.2, callbacks=[early_stopping], batch_size=1)

    # Evaluate the model on the test set
    loss = model.evaluate(x_test, y_test)
    print(f"Loss: {loss}")

    # Save the model and the scaler
    model.save('lstm_insulin_model.h5')
    dump(scaler, 'scaler_insulin.joblib')

def safe_strptime(date_string):
    for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%dT%H:%M:%SZ']:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    raise ValueError(f"time data '{date_string}' does not match any format")


def fetch_data(url, query, headers):
    response = requests.get(url, headers=headers, params=query)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}, Details: {response.text}")
        return None

def refresh_access_token():
    refresh_token = session.get('refresh_token')
    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    response = requests.post(TOKEN_URL, data=payload)
    if response.status_code == 200:
        token_info = response.json()
        session['access_token'] = token_info['access_token']
        session['expires_at'] = datetime.now() + timedelta(seconds=token_info['expires_in'])
    else:
        print("Failed to refresh token")

def check_refresh():
    current_time = datetime.now(timezone.utc)
    
    # Get the expiration time from the session and make it timezone-aware
    expires_at = session.get('expires_at')
    if expires_at is not None and expires_at.tzinfo is None:
        # If 'expires_at' is naive, make it aware by assuming it is in UTC
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    # Check if the access token needs to be refreshed
    if expires_at is not None and current_time >= expires_at:
        refresh_access_token()
    
    return {"Authorization": f"Bearer {session['access_token']}"}

@app.route('/login')
def login():
    state_value = secrets.token_urlsafe(16)
    session['oauth_state'] = state_value

    # Constructs the URL for the Dexcom OAuth 2.0 login page and redirects the user there
    query_params = {
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': SCOPE,
        'state': state_value
    }
    login_url = f"{AUTHORIZE_URL}?{urlencode(query_params)}"
    print(login_url)
    return redirect(login_url)

@app.route('/callback')
def callback():
    # Handles the callback from Dexcom after user authentication
    authorization_code = request.args.get('code')
    token_params = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': authorization_code,
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URI
    }
    # Exchanges the authorization code for an access token
    response = requests.post(TOKEN_URL, data=token_params)
    if response.status_code == 200:
        # Successfully obtained the access token
        access_token_info = response.json()
        session['access_token'] = access_token_info['access_token']
        session['refresh_token'] = access_token_info['refresh_token']
        session['expires_at'] = datetime.now() + timedelta(seconds=access_token_info['expires_in'])
        return redirect(url_for('train'))
    else:
        # Failed to obtain the access token
        return redirect(url_for('login'))
    # Make sure to store access_token, refresh_token, and expires_at in session

if __name__ == '__main__':
    app.run(debug=True, port=5001)