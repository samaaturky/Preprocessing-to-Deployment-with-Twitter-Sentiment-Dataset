from flask import Flask, request, jsonify
import joblib

# Load the saved pipeline
model = joblib.load("sentiment_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Sentiment Analysis API is running! Use /predict with POST."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    tweet = data.get("tweet", "")
    pred = model.predict([tweet])[0]
    return jsonify({"tweet": tweet, "predicted_sentiment": pred})

if __name__ == "__main__": 
    app.run(debug=True, use_reloader=False)
