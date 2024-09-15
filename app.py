from flask import Flask, request, render_template, url_for, redirect
import openai
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image  # Ensure PIL is imported
import os
from werkzeug.utils import secure_filename

# Load and preprocess the image
# Load the model
model = load_model('newmodel.h5')

classes = []
with open("assets/classes.txt") as f:
    for line in f.readlines():
        classes.append(line.split("\n")[0]) 

app = Flask(__name__)
UPLOAD_FOLDER = 'assets'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/saveAllergies', methods=['POST'])
def saveAllergies():
    allergies = request.form['allergies']
    with open("allergies.txt", "w") as f:
        f.write(allergies)
    return render_template('index.html')


def getAllergies():
    with open("allergies.txt", "r") as f:
        return f.read().split(',')


@app.route('/detectHazards', methods=['POST'])
def detectHazards():
    allergies = getAllergies()
    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
    file.save(filepath)

    img = tf.keras.utils.load_img(
    "assets/uploaded_image.png", target_size=(224, 224)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f}% percent confidence."
    .format(classes[np.argmax(score)], 100 * np.max(predictions[0]))
    )
    if 100 * np.max(predictions[0]) <= 50:
        return render_template('hazard_detection.html', hazard_content= "Unable to detect food. Please try again with a clearer image.")

    food = classes[np.argmax(score)]
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
        return render_template('hazard_detection.html', hazard_content= "WARNING: " + completion.choices[0].message['content'])
           


if __name__ == '__main__':
    app.run(debug=True, port=5000)