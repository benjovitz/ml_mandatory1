from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
app = Flask(__name__)

dataframe = pd.read_csv("./csv/customer_staying_or_not.csv")
x = dataframe.iloc[:, 3:13]
x = pd.get_dummies(x)
columnNames = list(x.columns)
x = x.values

scaler = StandardScaler()
x = scaler.fit_transform(x)

@app.route('/',methods=['post','get']) # will use get for the first page-load, post for the form-submit
def predict(): # this function can have any name
  try:
    model = load_model('customer_prediction_model.h5') # the mymodel.h5 file was created in Colab, downloaded and uploaded using Filezilla
    in1 = request.form.get('in1') # get the two numbers from the request object
    in2 = request.form.get('in2')
    in3 = request.form.get('in3')
    in4 = request.form.get('in4')
    in5 = request.form.get('in5')
    in6 = request.form.get('in6')
    in7 = request.form.get('in7')
    in8 = request.form.get('in8')
    in9 = request.form.get('in9')
    in10 = request.form.get('in10')
    in11 = request.form.get('in11')
    in12 = request.form.get('in12')
    in13 = request.form.get('in13')
    if in1 == None or in2 == None: # check if any number is missing
      return render_template('index.html', result='No input(s)')
        # calling render_template will inject the variable 'result' and send index.html to the browser
    else:
      arr = np.array([[float(in1), float(in2), float(in3), float(in4), float(in5), float(in6), float(in7), float(in8), float(in9), float(in10), float(in11), float(in12), float(in13)]]) # cast string to decimal number, and make 2d numpy array.
      arr = scaler.transform(arr)
      predictions = model.predict(arr) # make new prediction
      return render_template('index.html', result=str(predictions[0][0]))
        # the result is set, by asking for row=0, column=0. Then cast to string.
  except Exception as e:
    return render_template('index.html', result='error ' + str(e))

if __name__ == '__main__':
	app.run(host='0.0.0.0')
