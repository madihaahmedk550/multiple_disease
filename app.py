import numpy as np
import pickle as pk
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the saved models
cardiovascular_model = pk.load(open('cardiovascular_disease.sav', 'rb'))
stroke_model = pk.load(open('stroke_pred.sav', 'rb'))
diabetes_model = pk.load(open('diabetese_pred.sav', 'rb'))
liverdisease_model = pk.load(open('liver_disease.sav', 'rb'))

# Set up the page title
st.title("Multiple Disease Prediction Using ML")

# Input form

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("**Age**", min_value=0)
    
with col2:
    bmi = st.number_input("**BMI**", min_value=0)
       
    
with col1:
    cholesterol = st.selectbox("**Cholesterol**", ["Normal(below 200 mg/dL)", "Above Normal(between 200 - 239 mg/dL)", "Well Above Normal(above >= 240 mg/dL)"])
    cholesterol_code2 = {"Normal(below 200 mg/dL)":0,
                        "Above Normal(between 200 - 239 mg/dL)":  1,
                        "Well Above Normal(above >= 240 mg/dL)": 2}[cholesterol] 
    
with col2:
        glucose = st.selectbox("**Glucose**", ["Normal(below 100mg/dL)", "Above Normal(between 100mg/dL and 150mg/dL)", "Well Above Normal(150mg/dL or higher)"])
        glucose_code = {
            "Normal(below 100mg/dL)": 1,
            "Above Normal(between 100mg/dL and 150mg/dL)":0 ,
            "Well Above Normal(150mg/dL or higher)": 2
        }[glucose]
        
        
        glucose_code2 = {
            "Normal(below 100mg/dL)":0 ,
            "Above Normal(between 100mg/dL and 150mg/dL)":1 ,
            "Well Above Normal(150mg/dL or higher)": 2
        }[glucose]
        
        glucose_code3 = {
            "Normal(below 100mg/dL)":1 ,
            "Above Normal(between 100mg/dL and 150mg/dL)":2 ,
            "Well Above Normal(150mg/dL or higher)": 0
        }[glucose]
       
with col1:
     systolic_bp = st.selectbox("**Systolic Blood Pressure**",  ["Normal(below 120)", "Elevated( between 120 - 129)", "Hypertensive( above >= 130)"])
     systolic_bp_code2 = {
         "Normal(below 120)": 2,
         "Elevated( between 120 - 129)":1 ,
         "Hypertensive( above >= 130)": 0
     }[systolic_bp]
with col2:
     diastolic_bp = st.selectbox("**Diastolic Blood Pressure**", ["Normal(below 80)","Elevated( between 80 - 89)","Hypertensive( above >=90)"] )
     diastolic_bp_code2 = {
         "Normal(below 80)":2 ,
         "Elevated( between 80 - 89)" :  1,
         "Hypertensive( above >=90)":0
     }[diastolic_bp]
with col1:
        gender = st.radio("**Gender**", ["Male", "Female"])
        gender_code = 1 if gender == "Male" else 0

with col2:  
    ever_married = st.radio("**Ever Married**", ["No", "Yes"])
    ever_married_code = 1 if ever_married == "Yes" else 0
    
with col1:
    hypertension = st.radio("**Hypertension**", ["No", "Yes"])
    hypertension_code = 1 if hypertension == "Yes" else 0
with col2:
    heart_disease = st.radio("**Heart Disease**", ["No", "Yes"])
    heart_disease_code = 1 if heart_disease == "Yes" else 0
    heart_disease_code2 = 1 if heart_disease == "No" else 0
    

with col1:
    smoke = st.radio("**Smoke**", ["No","Yes"])
    smoke_code = 1 if smoke == "Yes" else 0
    smoke_code2 = 1 if smoke == "No" else 0    
with col2:
    alcohol = st.radio("**Alcohol Intake**", ["No", "Yes"])
    alcohol_code2 = 1 if alcohol == "No" else 0       


with col1:
     work_type = st.selectbox("**Work Type**", ["Private", "Self-employed", "Govt Job", "Never Worked", "Children"])
     work_type_code = {"Private": 2, "Self-employed": 3, "Govt Job": 0, "Never Worked": 1, "Children": 4}[work_type]
with col2:
     residence_type = st.selectbox("**Residence Type**", ["Rural", "Urban"])
     residence_type_code = 0 if residence_type == "Rural" else 1   
    

    
with col1:
     hba1c_level = st.selectbox("**HbA1c**", ["Normal(below 5.7%)", "Above Normal(between 5.7% and 6.4%)", "Well Above Normal(6.5% or higher)"]) 
     hba1c_level_code = {
         "Normal(below 5.7%)": 0,
         "Above Normal(between 5.7% and 6.4%)": 1,
         "Well Above Normal(6.5% or higher)": 2
     }[hba1c_level]
     
with col2:
    total_bilirubin = st.selectbox("**Total Bilirubin**", ["Normal (below 1.2 mg/dL)", "Severely Elevated (above 1.2 mg/dL)"])
    total_bilirubin_code = {
    "Normal (below 1.2 mg/dL)": 0,
    "Severely Elevated (above 1.2 mg/dL)": 1
}[total_bilirubin]

with col1:
   direct_bilirubin = st.selectbox("**Direct Bilirubin**", ["Normal (below 0.8 mg/dL)", "Severely Elevated (above 0.8 mg/dL)"])
   direct_bilirubin_code = {
    "Normal (below 0.8 mg/dL)": 0,
    "Severely Elevated (above 0.8 mg/dL)": 1
}[direct_bilirubin]
with col2:
# Alkaline Phosphatase (Alkphos) levels
  alkphos = st.selectbox("**Alkaline Phosphatase (Alkphos)**", ["Normal (below 100 U/L)", "Severely Elevated (above 100 U/L)"])
  alkphos_code = {
    "Normal (below 100 U/L)": 0,
    "Severely Elevated (above 100 U/L)": 1
}[alkphos]
with col1:
# SGPT_ALT (Alanine Aminotransferase) levels
  sgpt_alt = st.selectbox("**SGPT_ALT**", ["Normal (below 40 U/L)", "Severely Elevated (above 40 U/L)"])
  sgpt_alt_code = {
    "Normal (below 40 U/L)": 0,
    "Severely Elevated (above 40 U/L)": 1
}[sgpt_alt]
with col2:
# SGOT_AST (Aspartate Aminotransferase) levels
  sgot_ast = st.selectbox("**SGOT_AST**", ["Normal (below 40 U/L)", "Severely Elevated (above 40 U/L)"])
  sgot_ast_code = {
    "Normal (below 40 U/L)": 0,
    "Severely Elevated (above 40 U/L)": 1
}[sgot_ast]
with col1:
# Total Proteins levels
  total_proteins = st.radio("**Total_Proteins**", ["Normal (equal to or above 6.0 g/dL)","Low (below 6.0 g/dL)"])
  total_proteins_code = {
    "Low (below 6.0 g/dL)": 0,
    "Normal (equal to or above 6.0 g/dL)": 1
}[total_proteins]
with col2:
# ALB_Albumin (Albumin) levels
  albumin = st.radio("**ALB_Albumin**", ["Normal (above 3.5 g/dL)","Low (below 3.5 g/dL)"])
  albumin_code = {
    "Low (below 3.5 g/dL)": 0,
    "Normal (above 3.5 g/dL)": 1
}[albumin]
with col1:
# A/G_ratio (Albumin/Globulin ratio) levels
  ag_ratio = st.radio("**A/G_ratio**", ["Normal (above 1.0)","Low (below 1.0)"])
  ag_ratio_code = {
    "Low (below 1.0)": 0,
    "Normal (above 1.0)": 1,
}[ag_ratio]

with col2:
     physical_activity = st.radio("**Physical Activity**", ["Yes","No" ])
     physical_activity_code2 = 1 if physical_activity == "No" else 0   
    





# Prepare the input features
cvd_features = np.array([age, gender_code, bmi, glucose_code3, systolic_bp_code2, diastolic_bp_code2,cholesterol_code2, smoke_code, alcohol_code2, physical_activity_code2])
stroke_features = np.array([age, gender_code, bmi, glucose_code, hypertension_code, heart_disease_code,ever_married_code,work_type_code,residence_type_code,smoke_code2])
diabetes_features = np.array([age, gender_code, bmi, glucose_code2, hypertension_code, heart_disease_code2, hba1c_level_code,smoke_code])
liverdisease_features = np.array([age, gender_code, total_bilirubin_code, direct_bilirubin_code, alkphos_code, sgpt_alt_code, sgot_ast_code, total_proteins_code, albumin_code, ag_ratio_code])


# Predictions
if st.button("Predict"):
    if any([
        age is None or age == 0, bmi is None or bmi == 0, glucose is None or glucose == 0, 
    ]):
        st.warning("Please enter all the required input values and ensure they are non-zero.")
    else:
       
        cvd_prediction_proba = cardiovascular_model.predict_proba(cvd_features.reshape(1, -1))
        stroke_prediction_proba = stroke_model.predict_proba(stroke_features.reshape(1, -1))
        diabetes_prediction_proba = diabetes_model.predict_proba(diabetes_features.reshape(1, -1))
        liverdisease_prediction_proba = liverdisease_model.predict_proba(liverdisease_features.reshape(1, -1))
        
        
        cvd_probability = cvd_prediction_proba[0, 1] * 100
        stroke_probability = stroke_prediction_proba[0, 1] * 100
        diabetes_probability = diabetes_prediction_proba[0, 1] * 100
        liverdisease_probability = liverdisease_prediction_proba[0, 1] * 100
        
        
        
        st.markdown(f"### The possibility of the person having Cardio Vascular Disease is : {cvd_probability:.2f}%")
        st.markdown(f"### The possibility of the person having Brain Stroke Risk is: {stroke_probability:.2f}%")
        st.markdown(f"### The possibility of the person having Diabetes is: {diabetes_probability:.2f}%")
        st.markdown(f"### The possibility of the person having Liver Disease is: {liverdisease_probability:.2f}%")
        
        
        
# Bar graph data
        labels = ['Cardiovascular Disease', 'Brain Stroke Risk', 'Diabetes', 'Liver Disease']
        probabilities = [cvd_probability, stroke_probability, diabetes_probability, liverdisease_probability]

# Customize plot style
        sns.set_style("whitegrid")

# Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

# Define custom colors for the bars
        colors = ['lightblue', 'lightgreen', 'orange', 'lightpink']

# Plot the bar graph
        ax.bar(labels, probabilities, color=colors)

# Set x-axis and y-axis labels
        ax.set_xlabel('Disease', fontsize=12)
        ax.set_ylabel('Probability (%)', fontsize=12)

# Set plot title
        ax.set_title('Disease Predictions', fontsize=14, fontweight='bold')

# Add data labels on top of each bar
        for i, v in enumerate(probabilities):
            ax.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontsize=10)

# Remove spines (top and right) from the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# Display the plot in Streamlit
        st.pyplot(fig)
