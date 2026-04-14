# from flask import Flask, render_template, request

# app = Flask(__name__)

# # Dummy logic to predict deficiency
# def check_deficiency(symptoms):
#     symptoms = symptoms.lower()
#     if "fatigue" in symptoms or "pale" in symptoms:
#         return "Possible Vitamin B12 Deficiency"
#     elif "bone pain" in symptoms or "muscle weakness" in symptoms:
#         return "Possible Vitamin D Deficiency"
#     elif "night blindness" in symptoms or "dry eyes" in symptoms:
#         return "Possible Vitamin A Deficiency"
#     else:
#         return "Not enough information. Please consult a doctor."

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     result = ""
#     if request.method == 'POST':
#         user_symptoms = request.form['symptoms']
#         result = check_deficiency(user_symptoms)
#     return render_template('index.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import joblib

from flask_sqlalchemy import SQLAlchemy
import datetime

from flask import make_response, render_template_string
from xhtml2pdf import pisa



app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)


# Load model & vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        symptoms_vec = vectorizer.transform([symptoms])
        prediction = model.predict(symptoms_vec)
        result = f"🔎 Predicted deficiency: {prediction[0]}"
                # Save to DB
        new_prediction = Prediction(symptoms=symptoms, result=prediction[0])
        db.session.add(new_prediction)
        db.session.commit()
 

    return render_template('index.html', result=result)
@app.route('/history')
def history():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', records=records)

from flask import send_file
from io import BytesIO
from xhtml2pdf import pisa

@app.route('/download/<int:prediction_id>')
def download_pdf(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)

    # Render HTML string for PDF
    html = render_template_string("""
        <html>
        <head><title>Vitamin Report</title></head>
        <body>
            <h2>🩺 Vitamin Deficiency Report</h2>
            <p><strong>Date:</strong> {{ prediction.timestamp }}</p>
            <p><strong>Symptoms:</strong> {{ prediction.symptoms }}</p>
            <p><strong>Predicted Deficiency:</strong> {{ prediction.result }}</p>
        </body>
        </html>
    """, prediction=prediction)

    # Generate PDF into a memory buffer
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_buffer)

    if pisa_status.err:
        return "Error creating PDF", 500

    pdf_buffer.seek(0)  # move to beginning of file
    return send_file(pdf_buffer, download_name=f"report_{prediction_id}.pdf", as_attachment=True)



with app.app_context():
    db.create_all()


if __name__ == '__main__':
    app.run(debug=True)
