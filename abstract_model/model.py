# This is a abstract class that can be used by any ML model to perform specific tasks
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 

class ml_model:
    def __init__(self, X, y, model=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y) 
        self.model = model if model else RandomForestClassifier
        self.X = X
        self.y = y
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self):
        return self.model.predict(self.X_test)
    
    def print_results(self):
        prediction = self.predict()
        accuracy = accuracy_score(self.y_test, prediction)
        print(f"Model Accuracy: {accuracy:.2f}")
