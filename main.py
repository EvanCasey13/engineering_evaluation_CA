from subroutines.preprocessing.preprocessing_util import (
    data_selection,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Import model
from abstract_model.model import MLModel

# Get the dataframe we will use
dataframe = data_selection()
X = dataframe['x']

for col in ['t2', 't3', 't4']:
    le = LabelEncoder()
    dataframe[col] = le.fit_transform(dataframe[col])

# Define type variables
y_all_types = dataframe[['t2','t3','t4']]

# Initialize the model
ml_model = MLModel(X, y_all_types)

# vectorize 
X_train_transformed, X_test_transformed = ml_model.vectorize()

# Train the model using classifierchain
model_fit = ml_model.chain.fit(X_train_transformed, ml_model.y_train)

predictions = ml_model.chain.predict(X_test_transformed)

# convert both to numpy arrays to avoid error
y_test_np = np.array(ml_model.y_test) 
predictions_np = np.array(predictions)

# Print classification reports for each type
ml_model.print_classification_results()
