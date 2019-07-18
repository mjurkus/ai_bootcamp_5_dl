from flask import Flask, render_template, request
import tensorflow as tf
import pandas as pd

app = Flask(__name__)
model = tf.keras.models.load_model('model_dump.h5')


@app.route('/')
def hello():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    prediction = resolve_wine_quality(
        fixed_acidity=request.form['fixed_acidity'],
        volatile_acidity=request.form['volatile_acidity'],
        citric_acid=request.form['citric_acid'],
        residual_sugar=request.form['residual_sugar'],
        chlorides=request.form['chlorides'],
        free_sulfur_dioxide=request.form['free_sulfur_dioxide'],
        total_sulfur_dioxide=request.form['total_sulfur_dioxide'],
        density=request.form['density'],
        pH=request.form['pH'],
        sulphates=request.form['sulphates'],
        alcohol=request.form['alcohol'],
    )

    print("Prediction is {pre} " + str(prediction))

    return render_template('result.html', prediction=str(prediction))


def resolve_wine_quality(
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
) -> float:
    data = pd.DataFrame([fixed_acidity,
                         volatile_acidity,
                         citric_acid,
                         residual_sugar,
                         chlorides,
                         free_sulfur_dioxide,
                         total_sulfur_dioxide,
                         density,
                         pH,
                         sulphates,
                         alcohol])

    return model.predict(data.T)[0][0]


if __name__ == "__main__":
    app.run()
