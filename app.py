from flask import Flask, request, jsonify
from symptom import predict_disease
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Symptom-based Disease Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload with a "symptoms" key.
    Example: {"symptoms": ["itching", "skin_rash"]}
    """
    data = request.get_json()
    if not data or 'symptoms' not in data:
        return jsonify({'error': 'Invalid input. "symptoms" key is required.'}), 400

    symptoms = data['symptoms']
    
    try:
        disease, departments = predict_disease(symptoms)
        return jsonify({
            'predicted_disease': disease,
            'recommended_departments': departments
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
