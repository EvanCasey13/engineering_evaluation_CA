# This is a abstract class that can be used by any ML model to perform specific tasks
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import LabelEncoder

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
    
    def print_results(self):
        prediction = self.predict()
        accuracy = accuracy_score(self.y_test, prediction)
        print(f"Model Accuracy: {accuracy:.2f}")
