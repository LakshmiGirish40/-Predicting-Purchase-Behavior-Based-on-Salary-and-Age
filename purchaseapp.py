#User Interface File - Streamlit
import streamlit as st
import pickle
import numpy as np
import pandas as  pd
data = pd.read_csv(r"D:\Data_Science&AI\ClassRoomMaterial\dataset\logit classification.csv")

X = data[['Age','EstimatedSalary']]
#y dependent variable tarhet value
y = data['Purchased']
# Load the saved model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from  sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression() #Classifier is variable name
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# Set the title of the Streamlit app
st.title("purchased prediction")

# Add a brief description
st.write("This app predicts the salary and purchased based an age.")

# Add input widget for user to enter years of experience
Age= st.number_input("Enter Age", min_value=18, max_value=70,value=18, step=2)
Est_sal = st.slider("Enter Estimated salary",min_value=10000, max_value=200000,value=5000, step=2000)
st.markdown('The goal of this project is to develop a machine learning model that predicts whether a person will make a purchase based on their age and salary')
Gender = st.text("Select Male or Female")
st.checkbox("Male")
st.checkbox("Female")

input_data = np.array([[Age,Est_sal]])

# When the button is clicked, make predictions
if st.button("Purchase Prediction"):

    prediction = model.predict([[Age,Est_sal]])
    prediction_prob = model.predict_proba([[Age,Est_sal]])[0][1]
    
   
    st.write(f'The Age is: {Age}, the Salary is: {Est_sal} & Purchase Prediction is: {prediction[0]} with probability { prediction_prob:.2f}')
    st.write(f' The Estimated Salary is:  {Est_sal},  the Salary is:  {Est_sal} &  Purchase Prediction is:  {prediction[0]}  with probability  { prediction_prob:.2f}')
    # Display the result
    st.success(f"The Age is :{Age}, the salary is : {Est_sal} & purchase prediction is: {prediction[0]:,.2f}")
    
    
# Display information about the model
st.write("The model was trained using a dataset of Age and PurchasedEstimated Salary")

#streamlit run purchaseapp.py
