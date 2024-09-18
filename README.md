# AllergyPal
### Authors: Jacob Foster & Kyle Andrade

AllergyPal is a digital food safety aid designed to mitigate the risk of trying new foods.

A machine learning model trained on 100+ different dishes detects the name of your dish based on an image captured by the webcam. Then OpenAI determines
common ingredients in that food and cross references is it to the user's list of dietary restrictions. The user is then directed to a page
that notifies them of any risk.

### Prerequisites:
1. Install all packages using pip:
   - Flask
   - OpenAI (pip install openai==0.28)
   - Tensorflow
   - Numpy
   - Pillow
2. Obtain an OpenAI project API key and insert in app.py
  
