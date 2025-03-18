from app import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["answers"]

        # Convert text responses to numerical values (Modify based on your model)
        mapping = {
            "Not At All": 0,
            "Several Days": 1,
            "More Than Half The Days": 2,
            "Very Difficult": 3,
            "Somewhat Difficult": 2,
            "Not Difficult At All": 1
        }

        # Convert answers to numerical format
        numerical_input = [mapping.get(answer, 0) for answer in data]

        # Ensure model receives correct input format (Modify shape if needed)
        input_array = np.array(numerical_input).reshape(1, -1)

        # Get prediction (assuming binary classification: 0 or 1)
        prediction = model.predict(input_array)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
