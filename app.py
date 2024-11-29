import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Load the model and feature names
model_path = 'models/rfc_model.pkl'
with open(model_path, 'rb') as file:
    loaded_model, feature_names = pickle.load(file)

# Streamlit app title
st.title('Heart Disease Prediction')

st.write("""
## Enter the patient's information:
""")

# Input fields for all features with specified changes
def user_input_features():
    sex_options = {'Female': 0, 'Male': 1}
    chest_pain_options = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fasting_blood_sugar_options = {'No': 0, 'Yes': 1}
    resting_ecg_options = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    exercise_angina_options = {'No': 0, 'Yes': 1}
    st_slope_options = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}

    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.selectbox('Sex', options=list(sex_options.keys()))  
    chest_pain_type = st.selectbox('Chest Pain Type', options=list(chest_pain_options.keys()))  
    resting_bp_s = st.number_input('Resting Blood Pressure (systolic)', min_value=0, max_value=300, value=120)
    cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
    fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=list(fasting_blood_sugar_options.keys()))  
    resting_ecg = st.selectbox('Resting ECG', options=list(resting_ecg_options.keys()))  
    max_heart_rate = st.number_input('Max Heart Rate Achieved', min_value=0, max_value=250, value=150)
    exercise_angina = st.selectbox('Exercise Induced Angina', options=list(exercise_angina_options.keys()))  
    oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox('ST Slope', options=list(st_slope_options.keys())) 
    
    data = {
        'age': age,
        'sex': sex_options[sex],  
        'chest pain type': chest_pain_options[chest_pain_type],  
        'resting bp s': resting_bp_s,
        'cholesterol': cholesterol,
        'fasting blood sugar': fasting_blood_sugar_options[fasting_blood_sugar],  
        'resting ecg': resting_ecg_options[resting_ecg],  
        'max heart rate': max_heart_rate,
        'exercise angina': exercise_angina_options[exercise_angina],  
        'oldpeak': oldpeak,
        'ST slope': st_slope_options[st_slope]  
    }
    
    features = pd.DataFrame(data, index=[0])
    display_data = {
        'Age': age,
        'Sex': sex,
        'Chest Pain Type': chest_pain_type,
        'Resting BP Systolic': resting_bp_s,
        'Cholesterol': cholesterol,
        'Fasting Blood Sugar > 120 mg/dl': fasting_blood_sugar,
        'Resting ECG': resting_ecg,
        'Max Heart Rate Achieved': max_heart_rate,
        'Exercise Induced Angina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST Slope': st_slope
    }
    return features, display_data

# Get user input
input_df, display_data = user_input_features()

# Display the user input
st.write("### Patient's information:")
st.write(pd.DataFrame(display_data, index=[0]))

# Prediction button
if st.button('Predict'):
    # Make predictions
    prediction = loaded_model.predict(input_df)
    prediction_proba = loaded_model.predict_proba(input_df)

    # Display the prediction
    st.write("### Prediction")
    if prediction[0] == 1:
        st.write("Patient has heart disease.")
    else:
        st.write("Patient is healthy.")

    # Display the prediction probabilities
    st.write("### Prediction Probability")
    st.write(f"Probability of having heart disease: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of not having heart disease: {prediction_proba[0][0]:.2f}")
