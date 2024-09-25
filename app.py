from flask import Flask, request, render_template, url_for, redirect
import openai
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image  # Ensure PIL is imported
import os
from werkzeug.utils import secure_filename

## ADD OPENAI KEY HERE: openai.api_key = "your-api-key"

class AllergyPalApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = 'assets'
        self.model = load_model('newmodel.h5')
        self.classes = self.load_classes('assets/classes.txt')
        self.setup_routes()

    def load_classes(self, filepath):
        classes = []
        with open(filepath) as f:
            for line in f.readlines():
                classes.append(line.strip())
        return classes

    def setup_routes(self):
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/saveAllergies', 'saveAllergies', self.saveAllergies, methods=['POST'])
        self.app.add_url_rule('/detectHazards', 'detectHazards', self.detectHazards, methods=['POST'])

    def index(self):
        return render_template('index.html')

    def saveAllergies(self):
        allergies = request.form['allergies']
        with open("allergies.txt", "w") as f:
            f.write(allergies)
        return render_template('index.html')

    def getAllergies(self):
        with open("allergies.txt", "r") as f:
            return f.read().split(',')

    def detectHazards(self):
        allergies = self.getAllergies()
        file = request.files['image']
        filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
        file.save(filepath)

        img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
            "This image most likely belongs to {} with a {:.2f}% percent confidence."
            .format(self.classes[np.argmax(score)], 100 * np.max(predictions[0]))
        )
        if 100 * np.max(predictions[0]) <= 87:
            return render_template('hazard_detection.html', hazard_content="Unable to detect food. Please try again with a clearer image.")

        food = self.classes[np.argmax(score)]
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f" return any ingredients in {food} that are present in this list: {allergies}. Begin the list with 'This food contains the following allergens:' and then list. BE SURE TO LIST ALL. Otherwise print 'This food contains the following allergens: None.'"
                }
            ]
        )
        if completion.choices[0].message['content'] == "This food contains the following allergens: None.":
            return render_template('hazard_detection.html', hazard_content=completion.choices[0].message['content'])
        else:
            return render_template('hazard_detection.html', hazard_content="WARNING: " + completion.choices[0].message['content'])

    def run(self):
        self.app.run(debug=True, port=5000)

if __name__ == '__main__':
    app = AllergyPalApp()
    app.run()