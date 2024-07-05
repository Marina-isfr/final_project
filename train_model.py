import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from preprocess import fit_preprocess

df = pd.read_csv('/Users/marina/Desktop/final_project/data/cleaned/okcupid_preprocessed.csv')
# Preprocess the DataFrame
df, encoder, scaler = fit_preprocess(df)

# Define features (X) and target (y)
X = df.drop(['profile_id'], axis=1)  # Assuming 'profile_id' is not used in training
y = df['profile_id']  # Profile IDs will be used to identify similar profiles

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=20)

# Fit the k-NN model on the training data
knn.fit(X_train, y_train)

# Save the model, encoder, and scaler
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
