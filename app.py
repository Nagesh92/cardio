import numpy as np
from flask import Flask
from flask import render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    float_val = [float(x) for x in request.form.values()]
    final_val = [np.array(float_val)]

    prediction = model.predict(final_val)

    return render_template("new.html",data=prediction)


if __name__ == "__main__":
    app.run(debug=True)
