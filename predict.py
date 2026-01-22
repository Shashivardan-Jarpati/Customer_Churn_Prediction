from data_loader import load_data
from model import build_model

def predict_customer(sample):
    X, y = load_data()
    model = build_model()
    model.fit(X, y)

    prediction = model.predict([sample])

    return "❌ Customer will churn" if prediction[0] == 1 else "✅ Customer will stay"

if __name__ == "__main__":
    X, _ = load_data()
    sample_customer = X.iloc[0].values
    print(predict_customer(sample_customer))
