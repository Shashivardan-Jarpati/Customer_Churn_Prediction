from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from data_loader import load_data
from model import build_model

def train_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("✅ Model training completed")
    print("🎯 Accuracy:", accuracy)
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    train_model()
