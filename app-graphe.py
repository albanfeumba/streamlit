import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


st.title('My first app')

st.subheader("Auteur : Alban Feumba")

st.write(("Cette application renvoie l'histogramme d'une distribution normale"))

data=np.random.normal(size=1000)
data=pd.DataFrame(data, columns=["Dist_normal"])
#st.write(data.head())
st.dataframe(data.head())
#plt.hist(data.Dist_normal)
fig, ax = plt.subplots()
n_bins=st.number_input(
    label="Choisis un nombre de bins",
    min_value=10,
    value=20
    ) #permet à l'utilisateur d'entrer le nombre de bins et permettra d'avoir une application plus interactive
ax.hist(data.Dist_normal, bins=n_bins)
graph_title=st.text_input("Donnez un titre au graph:")
plt.title(graph_title)
st.pyplot(fig)
