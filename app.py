from flask import Flask, request, render_template
import openai


app = Flask(__name__)

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
    food = "spaghetti"
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
    return render_template('hazard_detection.html', hazard_content=completion.choices[0].message['content'])
           


if __name__ == '__main__':
    app.run(debug=True, port=5000)