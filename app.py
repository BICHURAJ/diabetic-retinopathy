import os
import requests
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv

load_dotenv()

URL= os.getenv('LOCAL_URL')
SERVER_URL=os.getenv('SERVER_URL')

def imageProcess(url):
    try:
        img = image.load_img(url, target_size=(224, 224))  # image to match model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension
        img_array = img_array / 255.0  # Normalize pixel values (assuming your model expects inputs in the range [0, 1])
        img_list = img_array.tolist()  # Convert NumPy array to Python list
        return json.dumps({'instances': img_list})
    except Exception as e:
        print("Error:", e)
        return None


filename= r'D:\artificial-intelligence\advanced-deep-learning-computer-vision\code\diabetic-retinopathy\data\raw\16_right.jpeg'
# the image
imageArray = imageProcess(filename)
class_labels = ['Level_0', 'Level_1', 'Level_2', 'Level_4']  

# Make a request to TensorFlow Serving
if imageArray is not None:
    headers = {'Content-Type': 'application/json'}
    response = requests.post(SERVER_URL, data=imageArray, headers=headers)
    predictions = response.json()['predictions'][0]
    predicted_class_index = np.argmax(predictions)
    print("Predictions:", class_labels[predicted_class_index])
else:
    print('Image Not Found.')
