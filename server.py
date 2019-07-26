import os
import base64
from io import BytesIO
from fastai import *
from fastai.vision import *
from flask import Flask, jsonify, request, render_template
from werkzeug.exceptions import BadRequest

def evaluate_image(img) -> str:
    pred_class, pred_idx, outputs = trained_model.predict(img)
    if pred_idx == 0:
        return 'car'
    elif pred_idx == 1:
        return 'animal'
    return ''

def load_model():
    path = Path('./data')
    learn = load_learner(path)
    return learn

app = Flask(__name__)
app.config['DEBUG'] = False
trained_model = load_model()

@app.route('/', methods=['GET'])
def index():
    return render_template('serving_template.html')

@app.route('/image', methods=['POST'])
def eval_image():
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File is not present in the request")
    if input_file.filename == '':
        return BadRequest("Filename is not present in the request")
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return BadRequest("Invalid file type")
    input_buffer = BytesIO()
    input_file.save(input_buffer)
    print(input_buffer)
    pred = evaluate_image(open_image(input_buffer))
    return jsonify({
        'prediction': pred,
    })

@app.route('/sample_image')
def eval_sample_image():
    pred = evaluate_image(open_image('./image.jpeg'))
    return jsonify({
        'prediction': pred
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=False)
