from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


# Load trained ML model
model = joblib.load(
    "model/accident_severity_model.pkl"
)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get data from frontend
        data = request.get_json()

        # Create DataFrame
        input_data = pd.DataFrame([data])


        # Features used during training
        features = [
            "Age_band_of_driver",
            "Sex_of_driver",
            "Driving_experience",
            "Type_of_vehicle",
            "Area_accident_occured",
            "Road_surface_type",
            "Road_surface_conditions",
            "Light_conditions",
            "Weather_conditions",
            "Type_of_collision",
            "Number_of_vehicles_involved",
            "Number_of_casualties",
            "Cause_of_accident"
        ]


        # Keep only required features
        input_data = input_data[features]


        # Convert numerical values
        input_data["Number_of_vehicles_involved"] = pd.to_numeric(
            input_data["Number_of_vehicles_involved"]
        )

        input_data["Number_of_casualties"] = pd.to_numeric(
            input_data["Number_of_casualties"]
        )


        # Make prediction
        prediction = model.predict(input_data)[0]


        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        classes = model.classes_


        probability_dict = {}

        for class_name, probability in zip(
            classes,
            probabilities
        ):

            probability_dict[str(class_name)] = round(
                float(probability) * 100,
                2
            )


        # Send result to frontend
        return jsonify({

            "success": True,

            "prediction": str(prediction),

            "probabilities": probability_dict

        })


    except Exception as error:

        print("ERROR:", error)

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


# Run Flask application
if __name__ == "__main__":

    app.run(debug=True)