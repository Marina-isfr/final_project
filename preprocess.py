import re
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Define categorical columns and keyword list
categorical_columns = ['status', 'sex', 'orientation', 'body_type', 'diet', 'drinks', 'drugs', 
                       'education', 'ethnicity', 'job', 'offspring', 'pets', 'religion', 
                       'sign', 'smokes', 'city']

keywords = ['love', 'fun', 'cooking', 'music', 'read', 'friends', 'work', 'travel', 'humor', 
            'movie', 'laugh', 'new', 'personality', 'future', 'people', 'smile', 'talk', 
            'life', 'tv', 'play']

# Function to create binary columns for keywords presence
def keyword_presence(text, keyword):
    pattern = re.compile(r'\b' + re.escape(keyword) + r'\w*\b', re.IGNORECASE)
    return 1 if pattern.search(text) else 0

# Function to fit and transform the preprocessing components
def fit_preprocess(df):
    # Create binary columns for each keyword in the long description columns
    columns_to_check = ['about_me', 'my_goals', 'my_talent', 'my_highlights', 'my_favorites',
                        'my_needs', 'think_about', 'typical_friday', 'my_secret', 'message_if']

    for keyword in keywords:
        df[keyword] = df[columns_to_check].apply(lambda row: any(keyword_presence(str(cell), keyword) for cell in row), axis=1)

    # Drop the original long description columns
    df = df.drop(columns_to_check, axis=1)

    # One-hot encode categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_columns = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop original categorical columns and add encoded columns
    df = df.drop(categorical_columns, axis=1)
    df = pd.concat([df, encoded_df], axis=1)

    # Convert keyword columns from boolean to integer
    df[keywords] = df[keywords].astype(int)

    # Normalize numeric columns
    scaler = MinMaxScaler()
    numeric_columns = ['age', 'height','income']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df, encoder, scaler

# Function to preprocess input data using fitted encoders and scalers
def preprocess_input(data, encoder, scaler):
    # Create binary columns for each keyword in the long description columns
    columns_to_check = ['about_me', 'my_goals', 'my_talent', 'my_highlights', 'my_favorites',
                        'my_needs', 'think_about', 'typical_friday', 'my_secret', 'message_if']

    for keyword in keywords:
        data[keyword] = data[columns_to_check].apply(lambda row: any(keyword_presence(str(cell), keyword) for cell in row), axis=1)

    # Drop the original long description columns
    data = data.drop(columns_to_check, axis=1)

    # One-hot encode categorical variables
    encoded_columns = encoder.transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop original categorical columns and add encoded columns
    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    # Convert keyword columns from boolean to integer
    data[keywords] = data[keywords].astype(int)

    # Normalize numeric columns
    numeric_columns = ['age', 'height','income']
    data[numeric_columns] = scaler.transform(data[numeric_columns])

    return data


