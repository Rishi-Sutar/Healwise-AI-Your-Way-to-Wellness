import numpy as np
import pandas as pd
import streamlit as st
import pickle
from src.utils import load_object
from src.exceptions import CustomException
from src.logger import logging
import sys

# Load model and data
svc = pickle.load(open('artifacts/model.pkl', 'rb'))

sym_des = pd.read_csv("artifacts/symtoms_df.csv")
precautions = pd.read_csv("artifacts/precautions_df.csv")
workout = pd.read_csv("artifacts/workout_df.csv")
description = pd.read_csv("artifacts/description.csv")
medications = pd.read_csv('artifacts/medications.csv')
diets = pd.read_csv("artifacts/diets.csv")

symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
    'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10,
    'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15,
    'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25,
    'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30,
    'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35,
    'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45,
    'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49,
    'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
    'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64,
    'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
    'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80,
    'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
    'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93,
    'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102,
    'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106,
    'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114,
    'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
    'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
    'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer disease',
    1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension', 30: 'Migraine',
    7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox',
    11: 'Dengue', 37: 'Typhoid', 40: 'Hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D',
    22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
    13: 'Dimorphic hemorrhoids (piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism',
    24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthritis', 5: 'Arthritis', 0: '(vertigo) Paroxysmal Positional Vertigo',
    2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

def helper(dis):
    try:
        desc = description[description['Disease'] == dis]['Description']
        if not desc.empty:
            desc = desc.values[0]
        else:
            desc = "No description available."

        pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        if not pre.empty:
            pre = pre.values[0]
        else:
            pre = ["No precautions available."]

        med = medications[medications['Disease'] == dis]['Medication']
        if not med.empty:
            med = med.values
        else:
            med = ["No medications available."]

        die = diets[diets['Disease'] == dis]['Diet']
        if not die.empty:
            die = die.values
        else:
            die = ["No dietary recommendations available."]

        wrkout = workout[workout['disease'] == dis]
        if not wrkout.empty:
            wrkout = wrkout['workout'].values
        else:
            wrkout = ["No workout recommendations available."]

        return desc, pre, med, die, wrkout
    except Exception as e:
        raise CustomException(e, sys)

def get_predicted_value(patient_symptoms):
    try:
        input_vector = np.zeros(len(symptoms_dict))
        for item in patient_symptoms:
            if item in symptoms_dict:
                input_vector[symptoms_dict[item]] = 1
                
        result = svc.predict([input_vector])[0]
        return result
    except Exception as e:
        raise CustomException(e, sys)

def main():
    try:
        st. set_page_config(layout="wide")
        html_temp = """
        <div style="background-color:blue;padding:10px">
        <h2 style="color:white;text-align:center;">Healwise: AI Your Way to Wellness 🩺</h2>
        </div>
        <br>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.header("Check Symptoms you have")
        # Convert symptoms dictionary keys to a list
        symptoms_list = list(symptoms_dict.keys())

        # Determine the split points for five columns
        num_symptoms = len(symptoms_list)
        split1 = num_symptoms // 5
        split2 = 2 * (num_symptoms // 5)
        split3 = 3 * (num_symptoms // 5)
        split4 = 4 * (num_symptoms // 5)

        # Split symptoms list into five parts
        symptoms_col1 = symptoms_list[:split1]
        symptoms_col2 = symptoms_list[split1:split2]
        symptoms_col3 = symptoms_list[split2:split3]
        symptoms_col4 = symptoms_list[split3:split4]
        symptoms_col5 = symptoms_list[split4:]

        # Create five columns
        col1, col2, col3, col4, col5 = st.columns(5)

        selected_symptoms = []

        # Add checkboxes to the first column
        with col1:
            for symptom in symptoms_col1:
                if st.checkbox(symptom):
                    selected_symptoms.append(symptom)

        # Add checkboxes to the second column
        with col2:
            for symptom in symptoms_col2:
                if st.checkbox(symptom):
                    selected_symptoms.append(symptom)

        # Add checkboxes to the third column
        with col3:
            for symptom in symptoms_col3:
                if st.checkbox(symptom):
                    selected_symptoms.append(symptom)

        # Add checkboxes to the fourth column
        with col4:
            for symptom in symptoms_col4:
                if st.checkbox(symptom):
                    selected_symptoms.append(symptom)

        # Add checkboxes to the fifth column
        with col5:
            for symptom in symptoms_col5:
                if st.checkbox(symptom):
                    selected_symptoms.append(symptom)
                    
        if st.button("Predict"):
            if selected_symptoms:
                predicted_disease = get_predicted_value(selected_symptoms)
                desc, pre, med, die, wrkout = helper(predicted_disease)

                st.success(predicted_disease)
                st.write("Description:", desc)
                st.write("Precautions:", '\n'.join([f"- {item}" for item in pre if pd.notna(item)]))
                st.write("Medicines:", ', '.join(med))
                st.write("Diet:", ', '.join(die))
                st.write("Workout:", ', '.join(wrkout))
            else:
                st.warning('Please select at least one symptom.')

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
