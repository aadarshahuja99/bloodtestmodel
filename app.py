from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd
from logging import FileHandler , WARNING
import pickle
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

sc=pickle.load(open('bpre.pickle','rb'))

clf = pickle.load(open('randomForestBloodTest.pickle','rb'))

app = Flask(__name__)
fh=FileHandler('errorlog.txt')
fh.setLevel(WARNING)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        nq = int(request.form['nq'])
        nm = int(request.form['nm'])
        a=int(request.form['age'])
        data=[nq,nm,a]
        # data=sc.transform(data)
        for i in range(len(data)):
            data[i]=int(data[i])
        vect = sc.transform(np.array([nq,nm,a]).reshape(1,-1))
        my_prediction = clf.predict(vect)
        print(my_prediction)
    return render_template('result.html',prediction = "{:.1f}".format(my_prediction[0]))
@app.route('/predict_api/')
def predict_api():
    nq = int(request.args['nq'])
    nm = int(request.args['nm'])
    a=int(request.args['age'])
    data=[nq,nm,a]
    for i in range(len(data)):
        data[i]=int(data[i])
    vect = sc.transform(np.array([nq,nm,a]).reshape(1,-1))
    my_prediction = clf.predict(vect)
    print(my_prediction)
    return jsonify({"prediction":"{:.1f}".format(my_prediction[0])})

#api.add_resource(home,'/')
#api.add_resource(predict_api,'/predict_api/<string:review>')

if __name__ == "__main__":
    app.run(debug=True)
