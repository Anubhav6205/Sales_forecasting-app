
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model=pickle.load(f);
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    values = [int(x) for x in request.form.values()]
    final_values=[np.array(values)]
    prediction=model.predict(final_values)
    
    output=round(prediction[0],2)
    return render_template('index.html',prediction='Sales can be about ${}'.format(output))
    
    

if __name__ == "__main__":
    app.run(debug=True)
