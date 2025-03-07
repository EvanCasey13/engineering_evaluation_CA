# This is a abstract class that can be used by any ML model to perform specific tasks
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer

class MLModel:
    def __init__(self, X, y, model=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
        self.model = model if model else RandomForestClassifier()
    
    def train_type_2(self, type):
        vectorizer = TfidfVectorizer()
        X_train_transformed = vectorizer.fit_transform(self.X_train)
        self.model.fit(X_train_transformed, type)
    
    def predict(self):
        return self.model.predict(self.X_test)
    
    def print_results(self):
        prediction = self.predict()
        accuracy = accuracy_score(self.y_test, prediction)
        print(f"Model Accuracy: {accuracy:.2f}")
