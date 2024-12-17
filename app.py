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

        # Open the image and transform it into a tensor
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image = image.resize((224, 224))  # Resize to model's expected input size
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = tf.convert_to_tensor([image_array])
        dataset = tf.data.Dataset.from_tensor_slices(image_tensor).batch(1)
        image_iterator = dataset.as_numpy_iterator()

        # Get the input data for the model
        input_data = next(image_iterator)

        # Make predictions
        predictions = model.predict(input_data).tolist()

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
