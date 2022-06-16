
from flask import Flask, render_template, url_for, request, jsonify      
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import csv
import numpy as np

app = Flask(__name__)

# Load the TF-IDF vocabulary specific to the category
with open("_pickles/Computer Science_vect.pkl", "rb") as f:
    cs = pickle.load(f)

with open("_pickles/Physics_vect.pkl", "rb") as f:
    phy = pickle.load(f)

with open("_pickles/Mathematics_vect.pkl", "rb") as f:
    maths = pickle.load(f)

with open("_pickles/Statistics_vect.pkl", "rb") as f:
    stat = pickle.load(f)

with open("_pickles/Quantitative Biology_vect.pkl", "rb") as f:
    qbio = pickle.load(f)

with open("_pickles/Quantitative Finance_vect.pkl", "rb") as f:
    qfin = pickle.load(f)

# Load the pickled RDF models
with open("_pickles/Computer Science_model.pkl", "rb") as f:
    cs_model = pickle.load(f)

with open("_pickles/Physics_model.pkl", "rb") as f:
    phy_model = pickle.load(f)

with open("_pickles/Mathematics_model.pkl", "rb") as f:
    maths_model  = pickle.load(f)

with open("_pickles/Statistics_model.pkl", "rb") as f:
    stat_model  = pickle.load(f)

with open("_pickles/Quantitative Biology_model.pkl", "rb") as f:
    qbio_model  = pickle.load(f)

with open("_pickles/Quantitative Finance_model.pkl", "rb") as f:
    qfin_model  = pickle.load(f)

# Render the HTML file for the home page
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    # Take a string input from user
    user_input = request.form['text']
    data = [user_input]

    vect = cs.transform(data)
    pred_cs = cs_model.predict_proba(vect)[:,1]

    vect = phy.transform(data)
    pred_phy = phy_model.predict_proba(vect)[:,1]

    vect = maths.transform(data)
    pred_maths = maths_model.predict_proba(vect)[:,1]

    vect = stat.transform(data)
    pred_stat = stat_model.predict_proba(vect)[:,1]

    vect = qbio.transform(data)
    pred_qbio = qbio_model.predict_proba(vect)[:,1]

    vect = qfin.transform(data)
    pred_qfin = qfin_model.predict_proba(vect)[:,1]

    out_cs = round(pred_cs[0], 2)
    out_phy = round(pred_phy[0], 2)
    out_maths = round(pred_maths[0], 2)
    out_stat = round(pred_stat[0], 2)
    out_qbio = round(pred_qbio[0], 2)
    out_qfin = round(pred_qfin[0], 2)

    F_list=[user_input,out_cs,out_phy,out_maths,out_stat,out_qbio,out_qfin]

    with open('predictions.csv','a') as csvfile:
        filewriter = csv.writer(csvfile)
        print("going to write")
        list=['Text','Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']
        filewriter.writerow(F_list)
        print("writing succesful")
    

    return render_template('home.html', 
                            pred_cs = 'Computer_Science: {}'.format(out_cs),
                            pred_phy = 'Physics: {}'.format(out_phy), 
                            pred_maths = 'Mathematics: {}'.format(out_maths),
                            pred_stat = 'Statistics: {}'.format(out_stat),
                            pred_qbio = 'Quantitative_Biology: {}'.format(out_qbio),
                            pred_qfin = 'Quantitative_Finance: {}'.format(out_qfin)                        
                            )
     
# Server reloads itself if code changes so no need to keep restarting:
app.run(debug=True)

