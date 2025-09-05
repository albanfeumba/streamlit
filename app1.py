import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Application de Machine Learning pour la detection de fraude par carte de credit")
    st.subheader("Auteur: Alban Feumba")
    
    #fonction d'importation des données
    @st.cache_data(persist=True)#Permet d'eviter que l'application soit lente
    def load_data(file):
        data=pd.read_csv(file)
        return data

    file=st.file_uploader("Charger un fichier csv" ,type=["csv"])
    if file is not None:
        df=load_data(file)
        df_sample=df.sample(5)
        if st.sidebar.checkbox("Afficher les Données brutes", False):
            st.subheader("Echantillon jeu de donnée creditcard")
            st.write(df_sample)
        #st.dataframe(data.head())#Affiche le dataframe du fichier telecharger

    
    #Affichage des données
#    df=load_data()
 #   df_sample=df.sample(5)
  #  if st.sidebar.checkbox("Afficher les Données brutes", False):
   #     st.subheader("Echantillon jeu de donnée creditcard")
    #    st.write(df_sample)
    
    seed=123
    #Train/Test split
    def split(df):
        y = df['Class']
        X = df.drop('Class', axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,#Permet de garder la meme distribution dans nos jeux de données
            random_state=seed
        ) 
        
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test=split(df)
    
    class_names=["T. Authentique", "T. Frauduleuse"]
    
    classifier = st.sidebar.selectbox(
        "Classificateur",
        ("Random Forest", "SVM", "Logistic Regression")
    )
    
    #Analyse de la performance des modèles
    def plot_perf(graphe):
        if "Confusion Matrix" in graphe:
            st.subheader("Matrice de confusion")
            fig1=ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,display_labels=class_names).figure_
            st.pyplot(fig1)
        
        if "ROC Curve" in graphe:
            st.subheader("Courbe ROC")
            fig2=RocCurveDisplay.from_estimator(model, X_test, y_test).figure_
            st.pyplot(fig2)
            
        if "Precision Recall Curve" in graphe:
            st.subheader("Precision Recall Courbe")
            fig3=PrecisionRecallDisplay.from_estimator(model, X_test, y_test).figure_
            st.pyplot(fig3)

    #Random Forest
    if classifier=="Random Forest":
        st.sidebar.subheader("Hyperparamètres du modèle")
        n_estimators=st.sidebar.number_input("Choisir le nombre de d'arbres dans la foret", 100, 1000, step=10, key="n_estimators")
        max_depth=st.sidebar.number_input("Profondeur maximale d'un arbre", 1, 20, step=1)
        #bootstrap=st.sidebar.radio("Echantillon bootstrap lors de la creation d'arbre", ("True","False"))
        bootstrap = st.sidebar.checkbox("Echantillon bootstrap lors de la creation d'arbre", value=True)
        
        graphes_perf=st.sidebar.multiselect("Choisir un graphique de performance du modèle", ("Confusion Matrix","ROC Curve","Precision Recall Curve"))
        
        if st.sidebar.button("Run",key="classify"):
            st.subheader("Random Forest Results")
            #Initialisation du modèle
            model=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, random_state=seed)
            model.fit(X_train,y_train)
            
            #Prediction
            y_pred=model.predict(X_test)
            
            #Metriques de performances
            accuracy=model.score(X_test, y_test)
            precision=precision_score(y_test,y_pred)
            recall=recall_score(y_test, y_pred)
            
            #Affichage des metriques
            st.write("Accuracy",accuracy)
            st.write("Precision",precision)
            st.write("Recall",recall)
            
            #Afficher les graphiques performances
            plot_perf(graphes_perf)
            
    #######################################################SVM##################################################################################
    if classifier=="SVM":
        st.sidebar.subheader("Hyperparamètres du modèle")
        hyp_c=st.sidebar.number_input("Choisir le paramètre de regularisation", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        kernel=st.sidebar.radio("Choisir le Kernel", ("rbf","linear","sigmoid","poly","precomputed"))
        gamma=st.sidebar.radio("Choisir le Gamma", ("scale","auto"))
        
        graphes_perf=st.sidebar.multiselect("Choisir un graphique de performance du modèle", ("Confusion Matrix","ROC Curve","Precision Recall Curve"))

        
        if st.sidebar.button("Run",key="classify"):
            st.subheader("SVM Results")
            #Initialisation du modèle
            model=SVC(C=hyp_c, kernel=kernel, gamma=gamma)
            model.fit(X_train,y_train)
            
            #Prediction
            y_pred=model.predict(X_test)
            
            #Metriques de performances
            accuracy=model.score(X_test, y_test)
            precision=precision_score(y_test,y_pred)
            recall=recall_score(y_test, y_pred)
            
            #Affichage des metriques
            st.write("Accuracy",accuracy)
            st.write("Precision",precision)
            st.write("Recall",recall)
            
            #Afficher les graphiques performances
            plot_perf(graphes_perf)
            
    #######################################################Regression logistique##################################################################################
    if classifier=="Logistic Regression":
        st.sidebar.subheader("Hyperparamètres du modèle")
        hyp_c=st.sidebar.number_input("Choisir le paramètre de regularisation", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        n_max_iter=st.sidebar.number_input("Nombre maximale d'iteration", 100, 1000, step=10)
        
        graphes_perf=st.sidebar.multiselect("Choisir un graphique de performance du modèle", ("Confusion Matrix","ROC Curve","Precision Recall Curve"))

        
        if st.sidebar.button("Run",key="classify"):
            st.subheader("Logistic Regression Results")
            #Initialisation du modèle
            model=LogisticRegression(C=hyp_c, max_iter=n_max_iter, random_state=seed)
            model.fit(X_train,y_train)
            
            #Prediction
            y_pred=model.predict(X_test)
            
            #Metriques de performances
            accuracy=model.score(X_test, y_test)
            precision=precision_score(y_test,y_pred)
            recall=recall_score(y_test, y_pred)
            
            #Affichage des metriques
            st.write("Accuracy",accuracy)
            st.write("Precision",precision)
            st.write("Recall",recall)
            
            #Afficher les graphiques performances
            plot_perf(graphes_perf)
            

if __name__=="__main__":
    main()

