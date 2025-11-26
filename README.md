# 🔍 Diabetic Retinopathy Detection (CapsNet + Flask)

Machine Learning system to detect diabetic retinopathy from retina images using **Capsule Networks (CapsNet)**, deployed via a **Flask web application** with **OpenCV-based validation** to prevent invalid image submissions.

---

## 🚀 Project Overview
- Achieved **91% accuracy** on test dataset using Capsule Network model.
- Built a **web-based interface** where users upload a retina scan to receive:
  - Predicted diagnosis (DR / No DR)
  - Probability score for each class
- Implemented **image validation using OpenCV** to reduce false predictions from non-retinal images.

---

## 🛠️ Tech Stack
| Component      | Technology Used |
|----------------|-----------------|
| ML Framework   | TensorFlow, NumPy |
| Web Deployment | Flask |
| Preprocessing  | OpenCV |
| Model Type     | Capsule Networks |
| Validation     | Retina shape detection |
| UI Template    | Bootstrap (FlexStart) |

---

## 📂 Folder Structure

Diabetic_retinopathy/
│── app.py # Flask application
│── model_utils.py # Model loading & inference helpers
│── image_validation.py # Retina image validation logic
│── CapsNet.Model/ # TensorFlow SavedModel
│── templates/ # HTML templates
│── static/ # CSS, JS, assets


---

## ▶️ How to Run Locally

```bash
git clone https://github.com/sujeeth-viswanathan/diabetic-retinopathy-ml.git
cd diabetic-retinopathy-ml
pip install -r requirements.txt
python app.py

Then go to → http://127.0.0.1:5000/

## 🧪 Future Improvements (Actively in Progress)

⌁ Integration of Grad-CAM for explainability

⬆ Retraining model with larger dataset

📄 Exportable PDF diagnostic reports

🌐 Deployment on cloud (Heroku/AWS)

## 📌 Author

👤 Sujeeth Viswanathan
Mechatronics & Computer Systems Engineer — passionate about AI/ML and embedded systems.

"AI should extend human capability, not replace it — this project focuses on early detection to prevent vision loss."
