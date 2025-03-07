from subroutines.preprocessing.preprocessing_util import (
    data_selection,
    data_model_preparation,
    handle_data_imbalance,
    remove_noise,
    display_results,
    text_data_rep,
    trans_to_en
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# Import the model
from abstract_model.model import MLModel

# Get the dataframe we will use
dataframe = data_selection()
X = dataframe['x']
# Define type variables
type_1 = dataframe['t1']
type_2 = dataframe['t2']
type_3 = dataframe['t3']
type_4 = dataframe['t4']

X_train, X_test, y_train, y_test = train_test_split(X, type_2, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Initialize the model
randomForest = RandomForestClassifier()

# Train the model
model = randomForest.fit(X_train_transformed, y_train)

predictions = model.predict(X_test_transformed)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)