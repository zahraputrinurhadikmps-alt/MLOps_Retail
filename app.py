from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("model_pembelian.pkl")


@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():

    sale_date = request.form["sale_date"]
    sale_time = request.form["sale_time"]
    gender = request.form["gender"]
    age = int(request.form["age"])
    category = request.form["category"]
    quantiy = int(request.form["quantiy"])
    price_per_unit = float(request.form["price_per_unit"])
    cogs = float(request.form["cogs"])

    data = pd.DataFrame([{
        "sale_date": sale_date,
        "sale_time": sale_time,
        "gender": gender,
        "age": age,
        "category": category,
        "quantiy": quantiy,
        "price_per_unit": price_per_unit,
        "cogs": cogs
    }])

    hasil = model.predict(data)

    return render_template(
        "index.html",
        prediction=round(float(hasil[0]),2),

        sale_date=sale_date,
        sale_time=sale_time,
        gender=gender,
        age=age,
        category=category,
        quantiy=quantiy,
        price_per_unit=price_per_unit,
        cogs=cogs
    )


if __name__ == "__main__":
    app.run(debug=True)