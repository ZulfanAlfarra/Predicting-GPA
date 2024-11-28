from flask import Flask, render_template, request
from src.components.data_processor import CustomData
from src.step.model_predicting_step import predict

app = Flask(__name__)

@app.route("/", methods= ['GET', 'POST'])
def predict_data():
    if request.method =='POST':
        data = CustomData(
            study_hours_per_day=request.form.get('study_hours'),
            extracurricular_hours_per_day=request.form.get('extracurricular_hours'),
            sleep_hours_per_day=request.form.get('sleep_hours'),
            social_hours_per_day=request.form.get('social_hours'),
            physical_activity_hours_per_day=request.form.get('physical_hours'),
            stress_level=request.form.get('stress')
        )

        df = data.get_df()

        pred = predict(df)[0]

        return render_template("index.html", prediction=pred)
    
    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)