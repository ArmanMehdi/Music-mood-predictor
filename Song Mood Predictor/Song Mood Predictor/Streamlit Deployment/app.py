
import streamlit as st 
import pandas as pd
import requests
import pickle
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Song Mood Predictor", page_icon=":headphones:", layout="wide")
# Function to load Lottie animation from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


col1, col2 = st.columns([1,4])
with col1:
    # Load Lottie animation for the app icon
    lottie_url = "https://lottie.host/58ab2799-0df1-427e-a744-1d081f92b15a/HQluYx5o8z.json"
    lottie_json = load_lottieurl(lottie_url)
    app_icon = st_lottie(lottie_json, speed=1, width=100, height=100, key="lottie")
with col2:
# Initial welcome message
    welcome_container = st.empty()
    welcome_container.header("Spotify Mood Recommender System")


    st.write("Enter the Values of the Song that you want, the model will classify the mood of song")


# Load the trained model
model_path = 'spotify_mood_model.pkl'  # Replace with the actual path to your model file
with open(model_path, 'rb') as model_file:
   model = pickle.load(model_file)

# Load the feature mapping for input features
input_features_path = 'input_features_mapping.pkl'  # Replace with the actual path to your input features file
with open(input_features_path, 'rb') as input_features_file:
   input_features_mapping = pickle.load(input_features_file)

# Define a dictionary mapping mood categories to emojis
mood_to_emoji = {
    'happy': '😃',
    'sad': '😢',
    'calm': '😌',
    'anger': '😠',
    'nostalgia': '🌟',
}


def get_numeric_input(prompt):
    user_input = st.text_input(prompt)

    # Check if the input is not empty and is a valid numeric value
    if user_input and user_input.replace('.', '', 1).isdigit():
        return float(user_input)
    else:
        print(f"Warning: Invalid numeric value entered for {prompt.replace('(numeric value)', '').strip()}.")
        return None


dance = get_numeric_input("Enter Dance (numeric value):")
energy = get_numeric_input("Enter Energy (numeric value):")
loudness = get_numeric_input("Enter Loudness (numeric value):")
valence = get_numeric_input("Enter Valence (numeric value):")
tempo = get_numeric_input("Enter Tempo (numeric value):")


# Two input modes (True or False)
input_mode = st.radio("Select Input Mode:", [True, False])

# Genre selection
genres = ['bollywood', 'classical', 'folk', 'ghazal', 'indie', 'pop', 'qawwali', 'reggae', 'rock', 'sufi']
selected_genre = st.selectbox("Select Genre:", genres)

# Range sliders
instrumentalness = st.slider("Instrumentalness:", 0.0, 1.0, step=0.1)
speechiness = st.slider("Speechiness:", 0.0, 1.0, step=0.1)
acousticness = st.slider("Acousticness:", 0.0, 1.0, step=0.1)
liveness = st.slider("Liveness:", 0.0, 1.0, step=0.1)


st.markdown("<br>", unsafe_allow_html=True)
# Button to trigger recommendations or further processing
if st.button("Predict Mood"):

    # Prepare the user input for the model
    user_input = pd.DataFrame({
        'dance': [dance],
        'energy': [energy],
        'loudness': [loudness],
        'valence': [valence],
        'tempo': [tempo],
        'input_mode': [input_mode],
        'selected_genre': [selected_genre],
        'instrumentalness': [instrumentalness],
        'speechiness': [speechiness],
        'acousticness': [acousticness],
        'liveness': [liveness],
    })

    st.markdown("<br>", unsafe_allow_html=True)
    # Map categorical features to numerical values based on your input_features_mapping
    # Example: user_input['selected_genre'] = input_features_mapping['selected_genre'][selected_genre]

    # Make predictions using the model
    predicted_mood = model.predict(user_input)[0]  # Assuming model.predict returns an array

   # Display the recommendations to the user along with the emoji
    emoji = mood_to_emoji.get(predicted_mood, '🤔')  # Default emoji for unknown mood
    st.success(f"Predicted Mood: {predicted_mood} {emoji}")


st.markdown(
    """
    <style>
        .footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background-color: #1e1e1e; /* Dark color */
        }
        .footer p {
            margin: 0;
            color: #fff; /* Text color */
        }
        .footer img {
            vertical-align: middle;
            margin-right: 10px;
        }
        .footer a {
            text-decoration: none;
            color: #fff; /* Link color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        <p>Made By <b>White Hat Team</b></p>
        <a href="https://github.com/your_username/your_repository" target="_blank">
            <img src="https://media.istockphoto.com/id/511653492/vector/incognito-hacker-spy-agent.jpg?s=612x612&w=0&k=20&c=IO9KG36TQ2wnIR3idSk-oEDV9zx5BsyduKQAlWPhNJU=" alt="White Hat Logo" width="30">
            Github Repository
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

