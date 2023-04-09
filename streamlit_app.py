import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
import pickle 
from pickle import dump

# define title bar
st.title('ModelDeployment: logistic regression')
st.sidebar.header('User input parameters')


# User diefined fucntion for accepting parameters


def user_input_features():
    CLMSEX = st.sidebar.selectbox('Gender',('1','0'))
    CLMINSUR = st.sidebar.selectbox('Insurace',('1','0'))
    SEATBELT = st.sidebar.selectbox('SeatBelt',('1','0'))
    CLMAGE = st.sidebar.number_input("Insert the Age")
    LOSS =  st.sidebar.number_input("Insert the LOSS")
    data = {'CLMSEX':CLMSEX,
            'CLMINSUR':CLMINSUR,
            'SEATBELT':SEATBELT,
            'CLMAGE':CLMAGE,
            'LOSS':LOSS}
    features = pd.DataFrame(data,index=[0])
    
    return features

df = user_input_features()
st.subheader("User Input Parameters")
st.write(df)


# Load the the Dataset

claimants= pd.read_csv("claimants.csv")
claimants.drop(["CASENUM"],inplace = True, axis =1)
claimants = claimants.dropna()
    
# Divide the dataset into X and Y

X = claimants.iloc[:,1:]
Y = claimants.iloc[:,0]


# Load and fit the data to the model   
clf = LogisticRegression()
clf.fit(X,Y)


#Validation

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)



st.subheader('Predicted result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)



#Save the model with pickle dump
pickle.dump(clf,open('first_weight.sav','wb'))
