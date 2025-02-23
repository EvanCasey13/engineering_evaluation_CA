from sklearn.ensemble import RandomForestClassifier

def model_selection(classifier):
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)

def train_model(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
