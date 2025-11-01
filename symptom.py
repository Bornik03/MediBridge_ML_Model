import joblib  
import numpy as np
import os

# Get the directory of the current script to build robust file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths to model files relative to the script's location
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer_symptom.pkl')
model_path = os.path.join(BASE_DIR, 'bornik_disease_prediction_model.pkl')
encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')

# Load models and encoders
model2 = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# All symptoms list
symptoms_list = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
    'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain',
    'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
    'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
    'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
    'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
    'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
    'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
    'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches',
    'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history',
    'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
    'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
    'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
    'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
    'red_sore_around_nose', 'yellow_crust_ooze'
]

# Disease-to-department mapping
disease_to_departments = {
    "Fungal Infection": ["General Medicine", "Infectious Diseases"],
    "Allergy": ["Allergy & Immunology"],
    "GERD": ["Gastroenterology"],
    "Chronic Cholestasis": ["Gastroenterology", "Hepatology"],
    "Drug Reaction": ["Allergy & Immunology", "Dermatology"],
    "Peptic Ulcer Disease": ["Gastroenterology"],
    "AIDS": ["Infectious Diseases", "HIV Clinic"],
    "Diabetes": ["Diabetology", "Endocrinology"],
    "Gastroenteritis": ["Gastroenterology"],
    "Bronchial Asthma": ["Pulmonology", "Allergy & Immunology"],
    "Hypertension": ["General Medicine", "Cardiology"],
    "Migraine": ["Neurology", "Headache Clinic"],
    "Cervical Spondylosis": ["Orthopedics", "Spine Specialist"],
    "Paralysis (brain hemorrhage)": ["Neurology", "Neurosurgery"],
    "Jaundice": ["Gastroenterology", "Hepatology"],
    "Malaria": ["Infectious Diseases", "General Medicine"],
    "Chickenpox": ["General Medicine", "Pediatrics"],
    "Dengue": ["Infectious Diseases", "General Medicine"],
    "Typhoid": ["Infectious Diseases", "General Medicine"],
    "Hepatitis A": ["Gastroenterology", "Hepatology"],
    "Hepatitis B": ["Gastroenterology", "Hepatology"],
    "Hepatitis C": ["Gastroenterology", "Hepatology"],
    "Hepatitis D": ["Gastroenterology", "Hepatology"],
    "Hepatitis E": ["Gastroenterology", "Hepatology"],
    "Alcoholic Hepatitis": ["Gastroenterology", "Hepatology"],
    "Tuberculosis": ["Pulmonology", "Infectious Diseases"],
    "Common Cold": ["General Medicine", "Family Medicine"],
    "Pneumonia": ["Pulmonology", "General Medicine"],
    "Dimorphic Hemorrhoids (piles)": ["General Surgery", "Gastroenterology"],
    "Heart Attack": ["Cardiology"],
    "Varicose Veins": ["Vascular Surgery", "General Surgery"],
    "Hypothyroidism": ["Endocrinology"],
    "Hypoglycemia": ["Endocrinology", "Diabetology"],
    "Osteoarthritis": ["Orthopedics", "Rheumatology"],
    "Arthritis": ["Rheumatology", "Orthopedics"],
    "Vertigo": ["Neurology", "ENT"],
    "Acne": ["Dermatology"],
    "Urinary Tract Infection": ["Urology", "General Medicine"],
    "Psoriasis": ["Dermatology"],
    "Impetigo": ["Dermatology"]
}

def predict_disease(symptoms):
    if not isinstance(symptoms, list) or not symptoms:
        raise ValueError("Invalid input. Provide a list of symptom names.")

    input_data = np.zeros(len(symptoms_list))
    for symptom in symptoms:
        if symptom in symptoms_list:
            idx = symptoms_list.index(symptom)
            input_data[idx] = 1
        else:
            raise ValueError(f"Invalid symptom '{symptom}' not found.")

    encoded = model2.predict([input_data])[0]
    disease = encoder.inverse_transform([encoded])[0].strip()

    if disease not in disease_to_departments:
        raise ValueError(f"No department mapping found for disease '{disease}'.")

    departments = disease_to_departments[disease]
    return disease, departments


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter symptoms separated by commas (or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        try:
            symptoms = [s.strip() for s in user_input.split(",")]
            disease, departments = predict_disease(symptoms)
            print(f"🩺 Predicted Disease: {disease}")
            print(f"🏥 Related Departments: {', '.join(departments)}")
        except Exception as e:
            print(f"⚠️ Error: {e}")
