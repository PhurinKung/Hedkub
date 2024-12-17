from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('classifierHedlek.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image file
        image_file = request.files['image']
        if not image_file:
            return jsonify({'error': 'No image uploaded'}), 400

        # Open the image and resize it to (224, 224) (model input size)
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image = image.resize((224, 224))  # Resize to model's expected input size
        
        # Convert to numpy array and normalize (scale pixel values to 0-1)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Expand dimensions to create a batch of size 1 (required for prediction)
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(image_array).tolist()

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
