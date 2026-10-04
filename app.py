from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)


application = Flask(__name__)

app = application


# ---------------------------------------------
# Home Page
# ---------------------------------------------

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ---------------------------------------------
# Prediction Page
# ---------------------------------------------

@app.route(
    "/predictdata",
    methods=["GET", "POST"]
)
def predict_datapoint():

    if request.method == "GET":

        return render_template(
            "home.html"
        )

    else:

        # -----------------------------------------
        # Get data from HTML form
        # -----------------------------------------

        data = CustomData(

            gender=request.form.get(
                "gender"
            ),

            race_ethnicity=request.form.get(
                "ethnicity"
            ),

            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),

            lunch=request.form.get(
                "lunch"
            ),

            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),

            # IMPORTANT:
            # Reading and writing were swapped
            # in your previous code.

            reading_score=float(
                request.form.get(
                    "reading_score"
                )
            ),

            writing_score=float(
                request.form.get(
                    "writing_score"
                )
            )
        )

        # -----------------------------------------
        # Convert input into DataFrame
        # -----------------------------------------

        pred_df = data.get_data_as_data_frame()

        print(
            "Input Data:"
        )

        print(
            pred_df
        )

        # -----------------------------------------
        # Prediction pipeline
        # -----------------------------------------

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(
            pred_df
        )

        # -----------------------------------------
        # Display prediction
        # -----------------------------------------

        return render_template(
            "home.html",
            results=results[0]
        )


# ---------------------------------------------
# Run Flask Application
# ---------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        debug=True
    )