from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# ------------------ Load Model ------------------
MODEL_PATH = r"C:\Users\vishwanath g\Desktop\cotton_disease_project\webapp\model\DenseNet121_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# ------------------ Classes ------------------
CLASS_NAMES = [
    'Bacterial Blight', 'Curl Virus', 'Healthy Leaf',
    'Herbicide Damage', 'Leaf Hopper Jassids',
    'Leaf Redding', 'Leaf Variegation'
]

# ------------------ Preprocess Function ------------------
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# ------------------ Routes ------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template('index.html', prediction="No file uploaded")

        # Save file
        os.makedirs('static', exist_ok=True)
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Predict
        img_array = preprocess_image(filepath)
        preds = model.predict(img_array)
        pred_class = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds) * 100

        return render_template(
            'index.html',
            prediction=pred_class,
            confidence=round(confidence, 2),
            img_path=filepath
        )

    # If GET (user just visited /predict)
    return render_template('index.html')

# ------------------ Run App ------------------
if __name__ == '__main__':
    app.run(debug=True)

