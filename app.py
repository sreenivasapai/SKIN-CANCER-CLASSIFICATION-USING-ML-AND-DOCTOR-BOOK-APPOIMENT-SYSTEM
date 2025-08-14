from flask import Flask, request, render_template, jsonify, redirect, session, make_response,url_for
import requests
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from datetime import datetime
import pdfkit
from bson.objectid import ObjectId
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import openai


# PDFKit config
config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# MongoDB connection
MONGO_URI = "mongodb+srv://project_pai_punk:1234567890@cluster0.im6dvhh.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["skin_cancer_db"]
users_collection = db["users"]
predictions_collection = db["predictions"]
appointments_collection = db["appointments"]

bcrypt = Bcrypt(app)

# Load Keras model
keras_model = load_model("static/final_skin_model.keras")
# Load skin image validator model
validator_model = load_model("static/lesion_validator2.keras")

def is_valid_skin_image(img_array):
    """
    Uses lesion_validator.keras to check if the uploaded image is a valid skin lesion.
    Returns True if valid, False if not.
    """
    from PIL import Image
    img_resized = Image.fromarray(img_array.astype(np.uint8)).resize((224, 224))
    img_array_resized = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_array_resized, axis=0)
    prediction = validator_model.predict(img_tensor)[0][0]
    return prediction > 0.5  # True = Valid skin image

class_names = ['Normal', 'Cancerous']

@app.route('/')
def root():
    return redirect('/login')

@app.route("/download-report/<patient_email>")
def download_report(patient_email):
    if 'username' not in session:
        return redirect('/login')

    # Mark appointments as viewed
    appointments_collection.update_many(
        {"patient_email": patient_email, "doctor_email": session['username'], "status": "Pending"},
        {"$set": {"status": "Viewed"}}
    )

    # Get patient and prediction
    patient = users_collection.find_one({"email": patient_email})
    prediction = predictions_collection.find_one({"user": patient_email}, sort=[('_id', -1)])
    if not patient or not prediction:
        return "Data not found", 404

    # Load image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], prediction['image'])
    if not os.path.exists(image_path):
        return "Image not found", 404

    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
        image_base64 = f"data:image/png;base64,{image_data}"

    # Convert confidence value safely
    confidence_value = prediction.get('confidence', 0)
    if isinstance(confidence_value, str) and confidence_value.endswith('%'):
        confidence_value = float(confidence_value.strip('%')) / 100
    else:
        confidence_value = float(confidence_value)

        raw_result = prediction.get("predicted_class", "")
        if str(raw_result).strip().lower() == "cancerous":
            prediction_result = "Cancerous"
        elif str(raw_result).strip().lower() == "normal":
            prediction_result = "Normal"
        else:
            prediction_result = "Normal"



    # Call AI Risk API
    try:
        risk_response = requests.post("http://127.0.0.1:5050/analyze", json={
            "prediction": prediction_result,
            "confidence": confidence_value
        })

        if risk_response.status_code == 200:
            risk_data = risk_response.json()
            graph_base64 = risk_data.get("graph_base64", "")
        else:
            risk_data = {
                "risk_level": "Unknown",
                "explanation": "Risk analysis service unavailable.",
                "precautions": [],
                "graph_base64": ""
            }
            graph_base64 = ""
    except Exception as e:
        print("Error calling risk analysis API:", e)
        risk_data = {
            "risk_level": "Unknown",
            "explanation": "Risk analysis service unavailable.",
            "precautions": [],
            "graph_base64": ""
        }
        graph_base64 = ""

    # Render HTML for PDF
    html = render_template("report_template.html",
                           patient=patient,
                           prediction=prediction,
                           image_data=image_base64,
                           risk=risk_data,
                           graph_base64=graph_base64,
                           date=datetime.now())

    # Convert to PDF
    pdf = pdfkit.from_string(html, False, configuration=config)
    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={patient.get('first_name', 'report')}_diagnosis_report.pdf"
    return response


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user_type = request.form.get('user_type')
        user = users_collection.find_one({'email': email, 'user_type': user_type})
        if user and bcrypt.check_password_hash(user['password'], password):
            session['username'] = email
            session['role'] = user_type
            return redirect('/doctor_dashboard' if user_type == 'doctor' else '/home')
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']
        user_data = {
            'email': email,
            'password': bcrypt.generate_password_hash(password).decode('utf-8'),
            'user_type': user_type,
            'first_name': request.form.get('first_name'),
            'last_name': request.form.get('last_name'),
            'phone': request.form.get('phone'),
            'address': request.form.get('address'),
            'dob': request.form.get('dob'),
            'gender': request.form.get('gender')
        }
        if user_type == 'doctor':
            user_data.update({
                'specialization': request.form.get('specialization'),
                'license': request.form.get('license'),
                'hospital': request.form.get('hospital')
            })
        if users_collection.find_one({'email': email, 'user_type': user_type}):
            return render_template('signup.html', error="Email already exists")
        users_collection.insert_one(user_data)
        return redirect('/login')
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'username' not in session or session.get('role') != 'patient':
        return redirect('/logout')
    user = users_collection.find_one({'email': session['username']})
    return render_template('home.html', username=session['username'], full_name=f"{user.get('first_name', '')} {user.get('last_name', '')}", prediction_result=session.get('prediction_result'))

@app.route('/doctor_dashboard')
def doctor_dashboard():
    if 'username' not in session or session.get('role') != 'doctor':
        return redirect('/logout')
    doctor_email = session['username']
    appointments = list(appointments_collection.find({'doctor_email': doctor_email}))
    doctor = users_collection.find_one({'email': doctor_email})
    return render_template(
        'doctor_dashboard.html',
        first_name=doctor.get('first_name', ''),
        last_name=doctor.get('last_name', ''),
        appointments=appointments,
        pending_count=appointments_collection.count_documents({'doctor_email': doctor_email, 'status': 'Pending'}),
        malignant_count=appointments_collection.count_documents({'doctor_email': doctor_email, 'status': 'Malignant'}),
        benign_count=appointments_collection.count_documents({'doctor_email': doctor_email, 'status': 'Benign'}),
        total_patients=len(set(app['patient_email'] for app in appointments))
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# Rule-based chatbot responses (extend as needed)
rule_based_responses = {
    "what is skin cancer": "Skin cancer is the abnormal growth of skin cells, usually due to UV exposure.",
    "what is melanoma": "Melanoma is a dangerous form of skin cancer that develops from pigment-producing cells.",
    "early signs of skin cancer": "Watch for changes in moles, new growths, or sores that don't heal.",
    "can skin cancer kill you": "Yes, if not treated early, certain types like melanoma can be fatal.",
    "is skin cancer itchy": "Some skin cancers may itch, bleed, or feel sore.",
    "how to prevent skin cancer": "Use sunscreen, avoid tanning beds, and wear protective clothing.",
    "what is a benign tumor": "A benign tumor is non-cancerous and doesn’t spread.",
    "what causes melanoma": "Excessive sun exposure and UV radiation are major causes.",
    "how to treat skin cancer": "Treatment may involve surgery, radiation, or topical medications.",
    "how do i know if my mole is dangerous": "If it changes in size, shape, or color, get it checked.",
    "is skin cancer curable": "Most skin cancers are curable if detected early and treated promptly.",
    "what are the types of skin cancer": "The main types are basal cell carcinoma, squamous cell carcinoma, and melanoma.",
    "who is at risk for skin cancer": "People with fair skin, history of sunburn, or family history of skin cancer are at higher risk.",
    "does skin cancer spread": "Yes, especially melanoma can spread to other parts of the body if not treated early.",
    "what does skin cancer look like": "It may appear as a new growth, a sore that doesn’t heal, or a changing mole.",
    "can children get skin cancer": "Yes, although rare, children can also develop skin cancer.",
    "is skin cancer contagious": "No, skin cancer is not contagious.",
    "does skin cancer only occur in fair-skinned people": "No, it can occur in all skin types, but is more common in fair-skinned individuals.",
    "how often should i check my skin": "It’s recommended to do a full skin self-exam once a month.",
    "can skin cancer appear suddenly": "Yes, some forms like melanoma can appear and spread rapidly.",
    "what is a skin biopsy": "A skin biopsy is a procedure where a small sample of skin is removed to check for cancer.",
    "does skin cancer hurt": "Not always. It can be painless, which is why regular checks are important.",
    "can sunscreen prevent skin cancer": "Sunscreen helps reduce the risk by protecting your skin from UV damage.",
    "what is basal cell carcinoma": "It’s the most common type of skin cancer and usually grows slowly.",
    "what is squamous cell carcinoma": "It’s a type of skin cancer that may grow faster and has a small chance to spread.",
    "can i get skin cancer on areas not exposed to the sun": "Yes, although rare, it can develop in non-exposed areas too.",
    "what is actinic keratosis": "It's a precancerous skin condition caused by sun damage that can lead to squamous cell carcinoma.",
    "should i be worried about a new mole": "Any new or changing mole should be evaluated by a dermatologist.",
    "how is skin cancer diagnosed": "Diagnosis typically involves physical exam and biopsy of the lesion.",
    "how is melanoma treated": "Melanoma treatment may include surgery, immunotherapy, targeted therapy, or chemotherapy.",
    "what is the abcde rule": "It helps detect melanoma: Asymmetry, Border, Color, Diameter >6mm, Evolving over time.",
    "can skin cancer return after treatment": "Yes, skin cancer can recur. Regular follow-up is important.",
    "what age group gets skin cancer most": "It's more common in people over 50, but younger individuals can also develop it.",
    "does tanning increase skin cancer risk": "Yes, both natural and artificial tanning increase the risk of skin cancer."
}

@app.route('/chatbot', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '').lower().strip()
        print("User input received:", user_input)  # Debug print

        response = "Sorry, I don't have an answer for that. Try asking something else."

        for key in rule_based_responses:
            if key in user_input:
                response = rule_based_responses[key]
                break

        return jsonify({'reply': response})
    
    except Exception as e:
        print("🔴 Chatbot Error:", str(e))  # Show error in terminal
        return jsonify({'reply': "Sorry, something went wrong."})

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect('/login')

    file = request.files.get('image')
    if not file or file.filename == '':
        return 'No image uploaded.', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess image
    image = Image.open(filepath).resize((300, 300)).convert("RGB")
    img_array_raw = np.array(image).astype(np.float32)

    # Skin Image Validation Step
    def is_valid_skin_image(img_array):
        img_resized = Image.fromarray(img_array.astype(np.uint8)).resize((224, 224))
        img_array_resized = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_array_resized, axis=0)
        prediction = validator_model.predict(img_tensor)[0][0]
        return prediction > 0.5

    if not is_valid_skin_image(img_array_raw):
        return render_template('home.html', result=None, username=session['username'], invalid_image=True)

    # Proceed to cancer prediction
    img_array = np.expand_dims(img_array_raw / 255.0, axis=0)
    prediction = keras_model.predict(img_array)[0][0]
    label = "Cancerous" if prediction >= 0.5 else "Normal"
    confidence = float(prediction if label == "Cancerous" else 1 - prediction)

    # Save prediction info
    session['prediction_result'] = {'predicted_class': label, 'confidence': confidence, 'image': filename}
    prediction_doc = {'user': session['username'], 'predicted_class': label, 'confidence': confidence, 'image': filename}
    session['last_prediction_id'] = str(predictions_collection.insert_one(prediction_doc).inserted_id)

    # Redirect if high-confidence cancerous
    if label == "Cancerous" and confidence > 0.60:
        doctors = list(users_collection.find({'user_type': 'doctor'}))
        from datetime import date
        return render_template('doctor_booking.html', username=session['username'], doctors=doctors, current_date=date.today())

    # Risk analysis only for borderline OR high-confidence normal
    show_risk_analysis = (0.45 <= confidence <= 0.55) or (label == "Normal" and confidence > 0.60) or (label == "Cancerous" and confidence<60)
    risk_data = {}

    if show_risk_analysis:
        try:
            response = requests.post("http://localhost:5050/analyze", json={
                "confidence": confidence,
                "prediction": label
            })
            if response.status_code == 200:
                risk_data = response.json()
            else:
                risk_data = {
                    "explanation": "Risk analysis service not available.",
                    "precautions": [],
                    "risk_level": "Unknown"
                }
        except:
            risk_data = {
                "explanation": "Risk analysis service not available.",
                "precautions": [],
                "risk_level": "Unknown"
            }

    return render_template(
        'home.html',
        result={'predicted_class': label, 'confidence': confidence},
        username=session['username'],
        show_risk=show_risk_analysis,
        explanation=risk_data.get('explanation'),
        precautions=risk_data.get('precautions'),
        risk_level=risk_data.get('risk_level')
    )



@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    if 'username' not in session:
        return redirect('/login')

    doctor_email = request.form.get('doctor_email')
    appointment_time = request.form.get('time')
    appointment_date = request.form.get('appointment_date')

    if not doctor_email or not appointment_time or not appointment_date:
        return 'Missing appointment details', 400

    # Get full patient name
    patient = users_collection.find_one({'email': session['username']})
    patient_name = f"{patient.get('first_name', '')} {patient.get('last_name', '')}"

    # Format the booked time
    booked_datetime = datetime.now().strftime('%B %d, %Y at %I:%M %p')

    appointment_data = {
        'doctor_email': doctor_email,
        'patient_email': session['username'],
        'patient_name': patient_name,
        'date': appointment_date,
        'time': appointment_time,
        'booked_datetime': booked_datetime,
        'status': 'Pending'
    }

    appointments_collection.insert_one(appointment_data)
    return redirect('/home')

@app.route('/book_appointment_page')
def book_appointment_page():
    if 'username' not in session:
        return redirect('/login')
    
    doctors = list(users_collection.find({'user_type': 'doctor'}))
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    return render_template('doctor_booking.html', 
                           username=session['username'], 
                           doctors=doctors,
                           current_date=current_date)

@app.route('/appointments')
def appointments():
    if 'username' not in session or session.get('role') != 'doctor':
        return redirect('/logout')
    return render_template('appointments.html', appointments=list(appointments_collection.find({'doctor_email': session['username']})), username=session['username'])

@app.route('/pending_appointments')
def pending_appointments():
    if 'username' not in session or session.get('role') != 'doctor':
        return redirect('/logout')

    doctor_email = session['username']
    
    # Fetch only pending appointments
    pending_apps = list(appointments_collection.find({'doctor_email': doctor_email, 'status': 'Pending'}))
    
    # Fetch full count for cards
    all_appointments = list(appointments_collection.find({'doctor_email': doctor_email}))
    doctor = users_collection.find_one({'email': doctor_email})

    return render_template(
        'doctor_dashboard.html',
        first_name=doctor.get('first_name', ''),
        last_name=doctor.get('last_name', ''),
        appointments=pending_apps,  # Filtered
        pending_count=appointments_collection.count_documents({'doctor_email': doctor_email, 'status': 'Pending'}),
        malignant_count=appointments_collection.count_documents({'doctor_email': doctor_email, 'status': 'Malignant'}),
        benign_count=appointments_collection.count_documents({'doctor_email': doctor_email, 'status': 'Benign'}),
        total_patients=len(set(app['patient_email'] for app in all_appointments)),
        active_section='pending'
    )

@app.route('/get_appointments')
def get_appointments():
    if 'username' not in session or session.get('role') != 'doctor':
        return jsonify([])

    doctor_email = session['username']
    appointments = appointments_collection.find({'doctor_email': doctor_email})

    events = []
    for appt in appointments:
        # Combine date and time into ISO format string
        appointment_datetime = f"{appt.get('date', '')}T{appt.get('time', '09:00')}"
        events.append({
            'title': appt.get('patient_name', 'Unknown'),
            'start': appointment_datetime,
            'extendedProps': {
                'email': appt.get('patient_email', 'N/A'),
                'status': appt.get('status', 'Pending'),
                'phone': appt.get('phone', 'N/A')
            }
        })

    return jsonify(events)

@app.route('/about')
def about():
    if 'username' not in session:
        return redirect('/login')
    return render_template('about_page.html', username=session['username'], doctors=list(users_collection.find({'user_type': 'doctor'})))

#Ensures only doctors can update statuses
@app.route('/update_status', methods=['POST'])
def update_status():
    if 'username' not in session or session.get('role') != 'doctor':
        return redirect('/logout')

    patient_email = request.form.get('patient_email')
    new_status = request.form.get('status')

    # Only allow update if current status is 'Pending'
    appointments_collection.update_one(
        {
            'doctor_email': session['username'],
            'patient_email': patient_email,
            'status': 'Pending'
        },
        {'$set': {'status': new_status}}
    )

    return redirect('/doctor_dashboard')

#Doctor Uploads a Case Image for AI Prediction + Risk Analysis
@app.route('/upload_case', methods=['POST'])
def upload_case():
    if 'username' not in session or session.get('role') != 'doctor':
        return redirect('/logout')

    file = request.files.get('image')
    if not file or file.filename == '':
        return 'No image uploaded.', 400

    # Ensure uploads folder exists
    upload_folder = os.path.join('static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Preprocess image for validation
    image = Image.open(filepath).resize((300, 300)).convert("RGB")
    img_array_raw = np.array(image).astype(np.float32)

    def is_valid_skin_image(img_array):
        img_resized = Image.fromarray(img_array.astype(np.uint8)).resize((224, 224))
        img_array_resized = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_array_resized, axis=0)
        prediction = validator_model.predict(img_tensor)[0][0]
        return prediction > 0.5

    if not is_valid_skin_image(img_array_raw):
        return render_template(
            'case_result.html',
            prediction_label="Invalid",
            confidence=0,
            risk_level="N/A",
            explanation="❌ This image is not a valid skin lesion. Please upload a proper dermoscopic image.",
            precautions=[],
            image_path=None,
            graph_path=None
        )

    # Proceed with normal prediction
    img_array = np.expand_dims(img_array_raw / 255.0, axis=0)
    prediction = keras_model.predict(img_array)[0][0]
    label = "Cancerous" if prediction >= 0.5 else "Normal"
    confidence = float(prediction if label == "Cancerous" else 1 - prediction)

    # Generate prediction graph
    import matplotlib.pyplot as plt
    labels = ["Normal", "Cancerous"]
    probabilities = [1 - prediction, prediction]
    graph_filename = f"doctor_graph_{filename}.png"
    graph_path = os.path.join(upload_folder, graph_filename)

    plt.figure(figsize=(4, 4))
    plt.bar(labels, probabilities, color=["green", "red"])
    plt.ylim(0, 1)
    plt.title("Prediction Confidence")
    plt.ylabel("Confidence Score")
    for i, v in enumerate(probabilities):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    # ✅ Call AI Risk Analysis API
    try:
        response = requests.post("http://127.0.0.1:5050/analyze", json={
            "prediction": label,
            "confidence": confidence
        })
        if response.status_code == 200:
            risk_data = response.json()
        else:
            raise Exception("Risk API error")
    except Exception as e:
        print("Risk API Error:", e)
        risk_data = {
            "risk_level": "Unknown",
            "explanation": "Risk analysis service not available.",
            "precautions": [],
        }

    # ✅ Pass everything to case_result.html
    return render_template(
        'case_result.html',
        prediction_label=label,
        confidence=round(confidence, 2),
        image_path=url_for('static', filename=f"uploads/{filename}"),
        graph_path=url_for('static', filename=f"uploads/{graph_filename}"),
        risk_level=risk_data.get("risk_level", "Unknown"),
        explanation=risk_data.get("explanation", "No explanation."),
        precautions=risk_data.get("precautions", [])
    )



# View a Previously Uploaded Case Result

@app.route('/case_result/<filename>')
def case_result(filename):
    if 'username' not in session or session.get('role') != 'doctor':
        return redirect('/logout')

    # Match full stored path
    prediction = predictions_collection.find_one({
        'image': f"uploads/{filename}",
        'user': session['username'],
        'uploaded_by': 'doctor'
    }, sort=[('timestamp', -1)])

    if not prediction:
        return "Prediction not found.", 404

    return render_template(
        'case_result.html',
        result=prediction,
        graph_path=url_for('static', filename=prediction['graph']),
        image_path=url_for('static', filename=prediction['image'])
    )

 
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

