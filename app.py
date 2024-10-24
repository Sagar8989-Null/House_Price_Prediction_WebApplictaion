from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import logging

app = Flask(__name__)

# Load the model
with open('CEP0 Model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up logging
logging.basicConfig(level=logging.DEBUG)


@app.route('/')
def home():
    return app.send_static_file('index.html')  # Ensure index.html is in a static folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    app.logger.debug(f"Received data: {data}")  # Log the received data

    try:
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([data])

        # Preprocess the input data using the loaded pipeline
        #processed_data = model.transform(input_df)

        # Make predictions
        prediction = model.predict(input_df)[0]

        return jsonify({'prediction': prediction})
    except Exception as e:
        app.logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)