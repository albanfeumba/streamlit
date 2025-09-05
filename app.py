import streamlit as st
import pandas as pd
import numpy as np

st.title("Application")

# Tracé linéaire
random_data=np.random.normal(size=1000)
st.line_chart(random_data)

# Diagramme à Barre
bar_data=pd.DataFrame(
    [100, 19, 88, 54],
    ["A", "B", "C", "D"]
)
st.bar_chart(bar_data)

# Carte 
df=pd.read_csv("AB_NYC_2019.csv").head(100)
st.write(df.head())
st.map(df[["longitude", "latitude"]])



