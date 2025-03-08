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

# Get the dataframe we will use
dataframe = data_selection()
X = dataframe['x']

for col in ['t2', 't3', 't4']:
    le = LabelEncoder()
    dataframe[col] = le.fit_transform(dataframe[col])

# Define type variables
type_1 = dataframe['t1']
type_2 = dataframe['t2']
type_3 = dataframe['t3']
type_4 = dataframe['t4']
y_all_types = dataframe[['t2','t3','t4']]

X_train, X_test, y_train, y_test = train_test_split(X, y_all_types, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

# Train the model using classifierchain
chain = ClassifierChain(rf_model, order='random', random_state=0)

model_fit = chain.fit(X_train_transformed, y_train)

predictions = chain.predict(X_test_transformed)

# convert both to numpy arrays to avoid error
y_test_np = np.array(y_test) 
predictions_np = np.array(predictions)

print(y_test_np.shape)
for i in range(y_test_np.shape[1]):
    print(f"classification report for output {i + 1}:")
    print(classification_report(y_test_np[:, i], predictions_np[:, i], zero_division=0))
