import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sys
import requests
from urllib.parse import urlencode

# Constants for Dexcom API OAuth 2.0 authentication
CLIENT_ID = 'Xv8e7QwMcm3jBHztPipV6tMEP6QFH4Zt'
CLIENT_SECRET = 'BnLfAZOnWVn2prSm'
REDIRECT_URI = 'http://127.0.0.1:5000'
AUTHORIZE_URL = 'https://api.dexcom.com/v2/oauth2/login'
TOKEN_URL = 'https://api.dexcom.com/v2/oauth2/token'
SCOPE = 'offline_access'

# Step One: Obtain Application Credentials
# Your application's credentials should be registered and obtained from the Dexcom developer portal.

# Step Two: Obtain User Authorization
# Direct the user to the Dexcom login page with the required query parameters
query_params = {
    'client_id': CLIENT_ID,
    'redirect_uri': REDIRECT_URI,
    'response_type': 'code',
    'scope': SCOPE,
}
login_url = f"{AUTHORIZE_URL}?{urlencode(query_params)}"
print(f"Navigate to the following URL to authorize: {login_url}")

# The user will interact with the Dexcom login page and authorize your application.
# After authorization, the user will be redirected to the redirect_uri with the authorization code in the query string.

# Step Three: Obtain Authorization Code
# This step is typically handled by your application's backend server or a web framework.
# The authorization code will be part of the query parameters in the redirect URI.
# You need to extract the 'code' parameter from the redirect URI.

# Example of extracting the authorization code from the redirect URI (not actual code execution):
# authorization_code = 'extracted_from_redirect_uri'

# Step Four: Obtain Access Token
# Exchange the authorization code for an access token using a POST request to the Dexcom token endpoint
AUTH_CODE = '3456cf0bdc117f347522834470c735b4'

def get_access_token(authorization_code):
    token_params = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': authorization_code,
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URI
    }
    response = requests.post(TOKEN_URL, data=token_params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to obtain access token: {response.content}")

# Example usage:
# Replace 'your_authorization_code' with the actual authorization code you received after user authorization
access_token_response = get_access_token(AUTH_CODE)
print(access_token_response)

sys.exit(0)

# Function to fetch data from Dexcom API
def fetch_data(url, query, headers):
    response = requests.get(url, headers=headers, params=query)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

# Replace <YOUR_TOKEN_HERE> with your actual Dexcom API token
headers = {"Authorization": "Bearer <YOUR_TOKEN_HERE>"}

# Fetch blood glucose values (egvs)
egvs_url = "https://api.dexcom.com/v3/users/self/egvs"
egvs_query = {
  "startDate": "2022-02-06T09:12:35",
  "endDate": "2022-02-06T09:12:35"
}
egvs_data = fetch_data(egvs_url, egvs_query, headers)

# Fetch event data
events_url = "https://api.dexcom.com/v3/users/self/events"
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