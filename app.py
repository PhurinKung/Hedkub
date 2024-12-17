from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('classifierHedlek.keras')

# Define image preprocessing function
def preprocess_image(img):
    # Resize the image to match the model's expected input size
    img = img.resize((256, 256))  # Resize image to 256x256 (adjust if necessary)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image to [0, 1] range
    return img_array

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image file
        image_file = request.files['image']
        if not image_file:
            return jsonify({'error': 'No image uploaded'}), 400
        
        # Open the image
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')

        # Preprocess the image
        img_array = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class] * 100
        
        # Return the prediction result
        return jsonify({
            'predicted_class': int(predicted_class),
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
