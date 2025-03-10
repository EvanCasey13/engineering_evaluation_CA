from subroutines.preprocessing.preprocessing_util import (
    data_selection,
)
from sklearn.preprocessing import LabelEncoder

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

# Print classification reports for each type
ml_model.print_classification_results()

# hamming loss
ml_model.get_hamming_loss()

# accuracy score
ml_model.get_accuracy_score()

# Precision score
ml_model.get_precision_score()

# Recall score
ml_model.get_recall_score()

# f1 score
ml_model.get_f1_score()