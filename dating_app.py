import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_input  # Import the preprocess function

# Load your DataFrame with all profile details
dfsi = pd.read_csv('/Users/marina/Desktop/final_project/data/cleaned/okcupid_preprocessed.csv')

# Load the saved k-NN model, encoder, and scaler
knn = joblib.load('knn_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Function to process user input and make predictions
def predict_similar_profiles(user_data):
    # Create a DataFrame from user input (assuming user_data is a dictionary)
    user_df = pd.DataFrame([user_data])

    # Perform preprocessing steps
    preprocessed_data = preprocess_input(user_df, encoder, scaler)

    # Find the nearest neighbors
    distances, indices = knn.kneighbors(preprocessed_data)
    return distances, indices

# CSS for background image
page_bg_img = '''
<style>
body {
    background-image: url("background.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''

# Inject CSS with the background image
st.markdown(page_bg_img, unsafe_allow_html=True)

# Main function to run the app
def main():
    # Initialize session state for page navigation and email submission
    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'
    if 'submitted_email' not in st.session_state:
        st.session_state.submitted_email = False

    # Welcome screen
    if st.session_state.page == 'welcome':
        st.title('Welcome to ❤️Todate❤️')
        st.header('You are just one click away to find your perfect match!')
        if st.button('Start'):
            st.session_state.page = 'input_form'

    # Input form screen
    elif st.session_state.page == 'input_form':
        st.subheader('Please enter your personal details and preferences')

        # Collect user input for profile attributes
        status = st.selectbox("Status", ['single', 'seeing someone', 'married'])
        sex = st.selectbox("Sex", ['m', 'f'])
        orientation = st.selectbox("Orientation", ['straight', 'bisexual', 'gay'])
        body_type = st.selectbox("Body Type", ['curvy', 'average', 'skinny', 'fit', 'full figure'])
        diet = st.selectbox("Diet", ['anything', 'other', 'vegetarian', 'rather not say', 'vegan'])
        drinks = st.selectbox("Drinks", ['sometimes', 'often', 'never', 'rather not say'])
        drugs = st.selectbox("Drugs", ['never', 'sometimes', 'rather not say', 'often'])
        education = st.selectbox("Education", ['College or more', 'Some college', 'High school or less', 'rather not say', 'Post graduate degree'])
        ethnicity = st.selectbox("Ethnicity", ['asian', 'white', 'hispanic', 'rather not say', 'other', 'black', 'indian'])
        job = st.selectbox("Job", ['transportation', 'hospitality', 'student', 'arts', 'it', 'finance', 'marketing', 'other', 'healthcare', 'media', 'science and tech', 'management', 'education', 'admin', 'construction', 'politics', 'legal', 'unemployed', 'military', 'retired'])
        offspring = st.selectbox("Offspring", ['no but want', 'no do not want', 'rather not say', 'yes'])
        pets = st.selectbox("Pets", ['likes both', 'likes cats', 'rather not say', 'likes dogs', 'dislikes both'])
        religion = st.selectbox("Religion", ['agnosticism', 'rather not to say', 'atheism', 'christianity', 'other', 'buddhism', 'judaism', 'hinduism', 'islam'])
        sign = st.selectbox("Sign", ['gemini', 'cancer', 'pisces', 'aquarius', 'taurus', 'sagittarius', 'leo', 'rather not to say', 'aries', 'libra', 'scorpio', 'virgo', 'capricorn'])
        smokes = st.selectbox("Smokes", ['sometimes', 'never', 'rather not say', 'often'])
        city = st.selectbox("City", ['south san francisco', 'oakland', 'berkeley', 'san francisco',
            'san mateo', 'daly city', 'san leandro', 'atherton', 'san rafael',
            'walnut creek', 'menlo park', 'belmont', 'san jose', 'palo alto',
            'emeryville', 'el granada', 'castro valley', 'fairfax',
            'mountain view', 'burlingame', 'martinez', 'alameda', 'vallejo',
            'benicia', 'mill valley', 'richmond', 'redwood city', 'el cerrito',
            'el sobrante', 'hayward', 'stanford', 'san pablo', 'novato',
            'pacifica', 'lafayette', 'half moon bay', 'fremont', 'orinda',
            'corte madera', 'san carlos', 'foster city', 'hercules',
            'santa cruz', 'san lorenzo', 'bolinas', 'sausalito', 'larkspur',
            'moraga', 'albany', 'san bruno', 'millbrae', 'petaluma', 'pinole',
            'pleasant hill', 'san geronimo', 'san anselmo', 'crockett',
            'freedom', 'belvedere tiburon', 'green brae', 'brisbane',
            'montara', 'ross', 'san quentin', 'hacienda heights', 'rodeo',
            'woodacre', 'westlake', 'rohnert park', 'sacramento',
            'point richmond', 'san diego', 'canyon country', 'west oakland',
            'kentfield', 'glencove', 'tiburon', 'east palo alto',
            'los angeles', 'hillsborough', 'union city', 'moss beach',
            'kensington', 'redwood shores', 'brea', 'woodside', 'lagunitas',
            'stinson beach', 'studio city', 'concord', 'piedmont', 'seaside',
            'forest knolls', 'magalia', 'colma', 'los gatos', 'sunnyvale',
            'santa monica', 'pasadena', 'arcadia', 'bayshore', 'milpitas',
            'port costa', 'nicasio', 'livingston', 'granite bay', 'isla vista',
            'hilarita', 'campbell', 'santa ana', 'santa rosa',
            'north hollywood', 'nevada city', 'stockton', 'marin city',
            'waterford', 'muir beach', 'pacheco', 'irvine', 'canyon',
            'oceanview', 'napa', 'san luis obispo', 'costa mesa', 'chico',
            'south lake tahoe', 'vacaville', 'long beach'])
        
        # Collect numeric input
        age = st.slider("Age", 18, 100)
        height = st.slider("Height (cm)", 140, 220)
        income = st.number_input("Income", value=0)

        # Example text inputs (optional)
        about_me = st.text_area("About Me")
        my_goals = st.text_area("My Goals")
        my_talent = st.text_area("My Talent")
        my_highlights = st.text_area("My Highlights")
        my_favorites = st.text_area("My Favorites")
        my_needs = st.text_area("My Needs")
        think_about = st.text_area("I Think About")
        typical_friday = st.text_area("Typical Friday")
        my_secret = st.text_area("My Secret")
        message_if = st.text_area("Message Me If")

        # Create a dictionary for user data
        user_data = {
            'status': status,
            'sex': sex,
            'orientation': orientation,
            'body_type': body_type,
            'city': city,
            'diet': diet,
            'drinks': drinks,
            'drugs': drugs,
            'education': education,
            'ethnicity': ethnicity,
            'job': job,
            'offspring': offspring,
            'pets': pets,
            'religion': religion,
            'sign': sign,
            'smokes': smokes,
            'age': age,
            'height': height,
            'income': income,
            'about_me': about_me,
            'my_goals': my_goals,
            'my_talent': my_talent,
            'my_highlights': my_highlights,
            'my_favorites': my_favorites,
            'my_needs': my_needs,
            'think_about': think_about,
            'typical_friday': typical_friday,
            'my_secret': my_secret,
            'message_if': message_if
        }

        # Find similar profiles button
        if st.button("Find Similar Profiles"):
            distances, indices = predict_similar_profiles(user_data)

            # Assuming dfsi is your original DataFrame containing all profiles
            similar_profiles_df = dfsi.iloc[indices.flatten()]

            # Display the DataFrame with details of similar profiles
            st.write("Details of similar profiles:")
            st.write(similar_profiles_df)

            # Subscription section
            st.markdown("---")
            st.write("To see the full profiles including pictures and stay up to date with events organized near you, please subscribe.")

            # Email input and submit button
            email = st.text_input("Please enter your email:")
            if st.button("Submit"):
                st.session_state.submitted_email = True
                st.session_state.page = 'thank_you'

    # Thank you screen
    elif st.session_state.page == 'thank_you' and st.session_state.submitted_email:
        st.write("Thank you, please check your inbox ❤️")
        
        # Button to go back to welcome screen
        if st.button("Back to Home"):
            st.session_state.page = 'welcome'
            st.session_state.submitted_email = False

if __name__ == '__main__':
    main()




