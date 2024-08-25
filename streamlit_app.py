import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

try: 
    # Title of the web app
    def load_data():
        st.title('Excel File Uploader')
        uploaded_file = st.file_uploader("Upload file", type=["csv"])
        return uploaded_file

    uploaded_file = load_data()
except:
    st.write('Upload file have problem')

if uploaded_file is not None:
    # Read the Excel file
    st.session_state.df = pd.read_csv(uploaded_file)

    columns_dict = {
    'one_hot': ['country','gender'], 
    'without_X': ['customer_id']
    }

    class Processing():
        def __init__(self, df: pd.DataFrame) -> None:
            self.df = df

        def one_hot(self, columns_dict: dict):
    
            self.df = pd.get_dummies(self.df, columns=columns_dict['one_hot'], drop_first=False)

            return self.df
        
        def feature_label(self, columns_dict: dict) -> pd.DataFrame:

            X = self.df.drop(columns = columns_dict['without_X'])

            return X


        def scale(self, X) -> pd.DataFrame:

            sc = StandardScaler()
            X_sc = sc.fit_transform(X)

            return X_sc

    try: 
        load_df = Processing(df = st.session_state.df)
        df = load_df.one_hot(columns_dict=columns_dict)
        X = load_df.feature_label(columns_dict=columns_dict)
        X_sc = load_df.scale(X)
    except: 
        st.write("Processing have problem")

    y_pred = model.predict(X_sc)
    y_pred_proba = model.predict_proba(X_sc)

    list_proba = list()
    for i in range(0, len(y_pred_proba)):
        if y_pred_proba[i][0] > y_pred_proba[i][1]:
            list_proba.append(y_pred_proba[i][0])
        else:
            list_proba.append(y_pred_proba[i][1])

    df_final = st.session_state.df
    df_final['y_proba'] = list_proba
    df_final['Post Prediction'] = 'Passed'

    option = st.selectbox(
    'Which your policy number?',
    df_final['customer_id'])

    score = df_final[df_final.customer_id == option]['y_proba']
    postpredict = df_final[df_final.customer_id == option]['Post Prediction']

    'Claim Score: ', round(score.iloc[0], 2)
    'Post Prediction:', postpredict.iloc[0]




