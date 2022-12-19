import numpy as np
import pickle
import streamlit as st
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('Adm_data1.pkl', 'rb'))

def run():
    # Adding title and Image
    img1 = Image.open('pngwing.com.png')
    img1 = img1.resize((156,145))
    st.image(img1,use_column_width=False)
    st.title("Univesity Admission Chances Prediction")

    # GRE Score
    Gre_score = st.number_input('GRE Score')

    # TOEFL Score
    TOEFL_Score = st.number_input('TOEFL Score')

    # University Rating
    University_Rating = st.number_input('University Rating (0-5)')

    # CGPA
    CGPA = st.number_input('CGPA')

    # Research
    Research = st.number_input('Have You Done Any Research?',min_value=0, max_value=1, step=1)
    

    if st.button('Submit'):
        features = [[Gre_score, TOEFL_Score, University_Rating, CGPA, Research]]
        print(features)
        prediction = model.predict(features)
        st.success(
                f'Your chances of getting addmission is {(prediction*100).round(2)}%'
            )
run()



