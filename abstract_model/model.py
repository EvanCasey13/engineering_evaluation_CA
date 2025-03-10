# This is a abstract class that can be used by any ML model to perform specific tasks
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, hamming_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
import numpy as np

class MLModel:
    def __init__(self, X, y, model=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
        self.model = model if model else RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        self.chain = ClassifierChain(base_estimator=self.model, order='random', random_state=0)
        
    def vectorize(self):
        vectorizer = TfidfVectorizer()
        self.X_train_transformed = vectorizer.fit_transform(self.X_train)
        self.X_test_transformed = vectorizer.transform(self.X_test)
        return self.X_train_transformed, self.X_test_transformed
        
    def predict(self):
        return self.chain.predict(self.X_test_transformed)
    
    def fit(self):
        # Train the model using classifierchain
        self.chain.fit(self.X_train_transformed, self.y_train)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)
    
    def print_classification_results(self):
        y_test_np = np.array(self.y_test) 
        predictions_np = np.array(self.predict())
        for i in range(y_test_np.shape[1]):
            print(f"classification report for output {i + 1}:")
            print(classification_report(y_test_np[:, i], predictions_np[:, i], zero_division=0))
            
    def get_hamming_loss(self):
        y_test_np = np.array(self.y_test) 
        predictions_np = np.array(self.predict())
        for i in range(y_test_np.shape[1]):
            print(f"hamming loss for output {i + 1}:")
            h_loss = hamming_loss(y_test_np[:, i], predictions_np[:, i])
            print(round(h_loss, 2))
        print()

    def get_accuracy_score(self):
        y_test_np = np.array(self.y_test) 
        predictions_np = np.array(self.predict())
        for i in range(y_test_np.shape[1]):
            print(f"accuracy score for output {i + 1}:")
            acc_score = accuracy_score(y_test_np[:, i], predictions_np[:, i])
            print(round(acc_score, 2))
        print()
            
    def get_precision_score(self):
        y_test_np = np.array(self.y_test) 
        predictions_np = np.array(self.predict())
        for i in range(y_test_np.shape[1]):
            print(f"precision score for output {i + 1}:")
            pre_score = precision_score(y_test_np[:, i], predictions_np[:, i], average='weighted', zero_division=0)
            print(round(pre_score, 2))
        print()
            
    def get_recall_score(self):
        y_test_np = np.array(self.y_test) 
        predictions_np = np.array(self.predict())
        for i in range(y_test_np.shape[1]):
            print(f"recall score for output {i + 1}:")
            rec_score = recall_score(y_test_np[:, i], predictions_np[:, i], average='weighted', zero_division=0)
            print(round(rec_score, 2))
        print()
        
    def get_f1_score(self):
        y_test_np = np.array(self.y_test) 
        predictions_np = np.array(self.predict())
        for i in range(y_test_np.shape[1]):
            print(f"f1 score for output {i + 1}:")
            f1 = f1_score(y_test_np[:, i], predictions_np[:, i], average='weighted', zero_division=0)
            print(round(f1, 2))
        print()