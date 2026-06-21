
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def make_data():
    np.random.seed(42)
    n=1200
    area=np.random.randint(700,5000,n)
    bedrooms=np.random.randint(1,6,n)
    age=np.random.randint(0,40,n)
    price=200000 + area*180 + bedrooms*15000 - age*2000 + np.random.normal(0,30000,n)
    return pd.DataFrame({"Area":area,"Bedrooms":bedrooms,"Age":age,"Price":price})

df=make_data()
X=df[["Area","Bedrooms","Age"]]
y=df["Price"]

model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X,y)

st.title("🏠 House Price Prediction (Synthetic Data)")

area=st.slider("Area (sq ft)",700,5000,2000)
bedrooms=st.slider("Bedrooms",1,6,3)
age=st.slider("Property Age",0,40,10)

pred=model.predict([[area,bedrooms,age]])[0]

st.metric("Predicted Price", f"₹{pred:,.0f}")
st.subheader("Training Data Sample")
st.dataframe(df.head())
