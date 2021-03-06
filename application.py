#!flask/bin/python
from flask import Flask
from flask import request
from flask import jsonify

import base64
import io
from PIL import Image, ImageChops
import figure_node
import dao

node = figure_node.FigureNode()
if not node.load():
    print('Create model')
    pre_fit_data = dao.get_pre_fit_data()
    node.pre_fit(pre_fit_data)
    dao.save_fit_data(pre_fit_data)
    print('Model created')
else:
    print('Model loaded')

application = Flask(__name__)


def crop(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    side = int(max(w, h) * 1.2)

    crop_img = im.crop(bbox)

    new_size = (side, side)
    result = Image.new("RGB", new_size, (255, 255, 255))
    result.paste(crop_img, (int((new_size[0]-w)/2), int((new_size[1]-h)/2)))

    return result


def get_image(data):
    decode = base64.b64decode(data)
    bio = io.BytesIO(decode)
    rgba = Image.open(bio)
    background = Image.new('RGBA', rgba.size, (255, 255, 255))
    im = Image.alpha_composite(background, rgba).convert('RGB')

    return crop(im)


@application.route('/')
def index():
    return "Hello! I'm a simple neural network."


@application.route('/predict', methods=('POST',))
def predict():
    data = request.get_data()

    prediction = node.predict(get_image(data))

    print('Prediction:', prediction)

    return jsonify(prediction), 200, {'Content-Type': 'application/json'}


@application.route('/fit', methods=('POST',))
def fit():
    data = request.get_data()
    label = request.args.get('label')

    print('Fit', label)

    fit_data = dao.get_fit_data()
    fit_data.append((get_image(data), label))
    node.fit(fit_data)

    dao.save_fit_data(fit_data)

    return 'success'


@application.route('/labels', methods=('GET',))
def labels():
    return jsonify(figure_node.LABELS)


if __name__ == '__main__':
    application.run(debug=False, threaded=True)
