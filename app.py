from flask import Flask, redirect, request, jsonify, make_response, session, url_for
import requests
from urllib.parse import urlencode
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timezone, timedelta
import secrets
from tqdm import tqdm

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
    return redirect(url_for('train'))

@app.route('/train')
def train():
    # Check if the user is logged in by checking the session
    if 'access_token' not in session:
        return redirect(url_for('login'))
    
    headers = check_refresh()
    
    range_url = "https://sandbox-api.dexcom.com/v3/users/self/dataRange"
    range_data = fetch_data(range_url, {}, headers)

    # Parse the start and end dates from the dataRange
    # start_date = safe_strptime(range_data['events']['start']['systemTime'])
    end_date = safe_strptime(range_data['events']['end']['systemTime'])
    start_date = end_date - timedelta(days=10)

    print(f"Start Date: {start_date}, End Date: {end_date}")

    egvs_df, events_df = fetch_and_process_data(start_date, end_date, headers)

    egvs_df.to_csv('egvs_data.csv', index=False)
    events_df.to_csv('events_data.csv', index=False)

    return "finished"

    # Example preprocessing steps
    # For simplicity, let's focus on 'value' from egvs and 'eventType' from events
    # More sophisticated feature engineering is required for a real-world application

    # Normalize the 'value' column in egvs_df
    scaler = MinMaxScaler(feature_range=(0, 1))
    egvs_df['normalized_value'] = scaler.fit_transform(egvs_df[['value']])

    # Map 'eventType' to numerical values in events_df
    event_type_mapping = {'carbs': 0, 'insulin': 1, 'exercise': 2}
    events_df['event_type_numeric'] = events_df['eventType'].map(event_type_mapping)

    # For this example, let's assume we're simplifying the problem to predict the next glucose value
    # based on the previous N values. A more complex model would also incorporate event data.

    # Function to create sequences for LSTM
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i+n_steps])
        return np.array(X), np.array(y)

    # Create sequences
    n_steps = 3  # Number of timesteps per sequence
    X, y = create_sequences(egvs_df['normalized_value'].values, n_steps)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Reshape X to fit the LSTM model
    X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

    # Train the model
    model.fit(X_reshaped, y, epochs=200, verbose=1)
    
    return 'Data fetched and processed successfully'

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
        
        mapping = {'control': 0, 'carbs': 1, 'exercise': 2}

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
                if overlapping_event['eventType'] in ['carbs', 'exercise']:
                    y_values = {'eventType': mapping[overlapping_event['eventType']]}
                else:
                    skip = False # Skip this interval if the event is not of interest
            else:
                # Interval does not overlap with any event, considered as control group
                y_values = {'eventType': mapping['control']}
            
            interval_egvs = [egv for egv in all_egvs_data.get('records', []) if current_interval_start <= safe_strptime(egv['systemTime']) < current_interval_end]
            x_values = [egv['value'] for egv in interval_egvs if (egv['value'] is not None) and skip]

            if x_values:  # Only include intervals with EGV data
                x_data.append(x_values[:6])
                y_data.append(y_values)

            # Move to the next interval
            current_interval_start = current_interval_end

    # Convert the lists to pandas DataFrames
    x_df = pd.DataFrame(x_data)
    y_df = pd.DataFrame(y_data)

    return x_df, y_df

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