from flask import Flask, render_template
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

app = Flask(__name__)
app.secret_key = 'juk'  # Change this to a random secret key

@app.route('/')
def index():
    return render_template('index.html')

# Constants for Dexcom API OAuth 2.0 authentication
CLIENT_ID = 'Xv8e7QwMcm3jBHztPipV6tMEP6QFH4Zt'
CLIENT_SECRET = 'rYww4k3RwxdYWPFo'
REDIRECT_URI = 'http://127.0.0.1:5001/callback'
AUTHORIZE_URL = 'https://sandbox-api.dexcom.com/v2/oauth2/login'
TOKEN_URL = 'https://sandbox-api.dexcom.com/v2/oauth2/token'
SCOPE = 'offline_access'


@app.route('/before')
def routeBeforeLogin():
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

    x_new = [1, 2, 3, 4, 5]
    y_new = [1, 4, 9, 16, 25]

    # Check if the form has been submitted
    if request.method == 'POST':
        basil_rate = request.form['basilRate']
        resp = make_response(render_template('index.html', x_new=x_new, y_new=y_new))
        resp.set_cookie('basilRate', basil_rate)
        return resp

    # Attempt to read the basilRate from the cookie
    basil_rate = request.cookies.get('basilRate', 'Not set')

    return render_template('user.html', x_values=x_new, y_values=y_new, meal_occured=False, insulin_needed = 10, yesno="Yes")

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
        return redirect(url_for('routeBeforeLogin'))
    else:
        # Failed to obtain the access token
        return redirect(url_for('login'))
    # Make sure to store access_token, refresh_token, and expires_at in session

if __name__ == '__main__':
    app.run(port=5001,debug=True) 