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
    # Check if the user is logged in by checking the session
    if 'access_token' not in session:
        return redirect(url_for('login'))
    
    # Get the current time in UTC
    current_time = datetime.now(timezone.utc)
    
    # Get the expiration time from the session and make it timezone-aware
    expires_at = session.get('expires_at')
    if expires_at is not None and expires_at.tzinfo is None:
        # If 'expires_at' is naive, make it aware by assuming it is in UTC
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    # Check if the access token needs to be refreshed
    if expires_at is not None and current_time >= expires_at:
        refresh_access_token()
    
    headers = {"Authorization": f"Bearer {session['access_token']}"}
    
    # Fetch data range
    range_url = "https://sandbox-api.dexcom.com/v3/users/self/dataRange"
    range_data = fetch_data(range_url, {}, headers)

    egvs_url = "https://sandbox-api.dexcom.com/v3/users/self/egvs"
    egvs_query = {
    "startDate": str(range_data['egvs']['start']['systemTime']),
    "endDate": str(range_data['egvs']['end']['systemTime'])
    }
    egvs_data = fetch_data(egvs_url, egvs_query, headers)

    return str(egvs_data)

    # Fetch event data
    events_url = "https://sandbox-api.dexcom.com/v3/users/self/events"
    events_query = {
    "startDate": "2023-01-01T09:12:35",
    "endDate": "2023-01-01T09:12:35"
    }
    events_data = fetch_data(events_url, events_query, headers)

    # Assuming the fetched data is stored in egvs_data and events_data variables
    # Convert the data to pandas DataFrame for easier manipulation
    egvs_df = pd.DataFrame(egvs_data['records'])
    events_df = pd.DataFrame(events_data['records'])

    # Example preprocessing steps
    # For simplicity, let's focus on 'value' from egvs and 'eventType' from events
    # More sophisticated feature engineering is required for a real-world application

    # Normalize the 'value' column in egvs_df
    scaler = MinMaxScaler(feature_range=(0, 1))
    egvs_df['normalized_value'] = scaler.fit_transform(egvs_df[['value']])

    # Map 'eventType' to numerical values in events_df
    event_type_mapping = {'carbs': 0, 'shortActing': 1, 'longActing': 2, 'exercise': 3}
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

def fetch_data(url, query, headers):
    response = requests.get(url, headers=headers, params=query)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
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
        return redirect(url_for('index'))
    else:
        # Failed to obtain the access token
        return redirect(url_for('login'))
    # Make sure to store access_token, refresh_token, and expires_at in session

if __name__ == '__main__':
    app.run(debug=True, port=5001)