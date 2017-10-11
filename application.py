#!flask/bin/python
from flask import Flask
from flask import request
from flask import jsonify

import base64
import io
from PIL import Image, ImageChops
import figure_node
import dao

app = Flask(__name__)


def crop(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    side = max(w, h) * 1.2
    w_ext = int((side - w) / 2)
    h_ext = int((side - h) / 2)

    x1 = max(0, bbox[0] - w_ext)
    y1 = max(0, bbox[1] - h_ext)

    x2 = min(im.size[0], bbox[2] + w_ext)
    y2 = min(im.size[1], bbox[3] + h_ext)

    return im.crop((x1, y1, x2, y2))


def get_image(data):
    decode = base64.b64decode(data)
    bio = io.BytesIO(decode)
    rgba = Image.open(bio)
    background = Image.new('RGBA', rgba.size, (255, 255, 255))
    im = Image.alpha_composite(background, rgba).convert('RGB')

    return crop(im)


@app.route('/')
def index():
    return "Hello! I'm a simple neural network."


@app.route('/predict', methods=('POST',))
def predict():
    data = request.get_data()
    print('Data to predict:', data)
    return jsonify(node.predict(get_image(data))), 200, {'Content-Type': 'application/json'}


@app.route('/fit', methods=('POST',))
def fit():
    data = request.get_data()
    label = request.args.get('label')

    fit_data = dao.get_fit_data()
    fit_data.append((get_image(data), label))
    node.fit(fit_data)

    dao.save_fit_data(fit_data)

    return 'success'


@app.route('/labels', methods=('GET',))
def labels():
    return jsonify(figure_node.LABELS)


if __name__ == '__main__':
    node = figure_node.FigureNode()
    if not node.load():
        print('Create model')
        pre_fit_data = dao.get_pre_fit_data()
        node.fit(pre_fit_data)
        dao.save_fit_data(pre_fit_data)
        print('Model created')
    else:
        print('Model loaded')

    app.run(debug=False)
