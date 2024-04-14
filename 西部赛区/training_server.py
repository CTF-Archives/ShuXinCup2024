import os
import random
import hashlib
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.optimizers import Adamax

from secret import flag

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['PERMANENT_SESSION_LIFETIME'] = 600

@app.route('/')
def index():
    return "<h1>I'm alive.</h1>"

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global loaded_model
    try:
        if 'model' not in request.files:
            return jsonify({'status': 'failed', 'message': 'No model file.'})
        model = request.files['model']
        model.save('./model.h5')
        loaded_model = tf.keras.models.load_model('./model.h5', compile=False)
        loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
        with open('./model.h5', 'rb') as f:
            model_bytes = f.read()
        return jsonify({'status': 'success', 'message': hashlib.md5(model_bytes).hexdigest()})
    except Exception as e:
        loaded_model = None
        # print(repr(e))
        return jsonify({'status': 'failed', 'message': repr(e)})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not loaded_model:
        return jsonify({'status': 'failed', 'message': 'No model.'})
    counts = 50
    right_count = 0
    fire_count = random.randint(1, counts // 2)
    no_fire_count = counts - fire_count
    fire_names = random.sample(FireNames, fire_count)
    no_fire_names = random.sample(NoFireNames, no_fire_count)

    for name in fire_names:
        image = Image.open('./Vali/Fire/' + name)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = loaded_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        if tf.argmax(score).numpy() == 0:
            right_count += 1
    for name in no_fire_names:
        image = Image.open('./Vali/NoFire/' + name)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = loaded_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        if tf.argmax(score).numpy() == 1:
            right_count += 1
    
    if right_count / counts > 0.9:
        return jsonify({'status': 'success', 'right': f'{right_count/counts:0.2f}%', 'message': flag})
    else:
        return jsonify({'status': 'failed', 'right': f'{right_count/counts:0.2f}%',  'message': 'Sorry, you are not good enough.'})

if __name__ == '__main__':
    loaded_model = None
    FireNames = os.listdir('./Vali/Fire')
    NoFireNames = os.listdir('./Vali/NoFire')

    app.run(host='0.0.0.0', port = 5000, debug=False)

