from keras import backend as K
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import matplotlib
from keras import optimizers, initializers
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, \
    Flatten, UpSampling2D, GaussianNoise, \
    Reshape, Lambda, Add, concatenate, Conv2DTranspose, \
    Dropout, LeakyReLU, Activation, GlobalAveragePooling2D, PReLU, Softmax, Multiply
from keras.models import model_from_json, Model
import UtilsFunction as UF
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler, EarlyStopping, \
    ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import History
from model import Deeplabv3
import tensorflow as tf

#matplotlib.use('TkAgg')
from matplotlib import gridspec
from matplotlib import pyplot as plt
from flask import Flask
from flask_restful import Resource, Api
from PIL import Image
import numpy as np
import flask
import requests
import json
import io

import os

import UMask.UMask as U

import sys
sys.stdout.flush()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


app = Flask(__name__)
model = None

# ---------------------------------------------------------------
#                             Losses
# ---------------------------------------------------------------
def iou_loss(y_true, y_pred, smooth=10):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def sym_dif(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.sum(y_true_f) + K.sum(y_pred_f) - 2*intersection

# ---------------------------------------------------------------
#                             Model
# ---------------------------------------------------------------
def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model

    model = Deeplabv3(input_shape=(256, 256, 3),
                      classes=1,
                      weights='cityscapes',
                      activation='sigmoid',
                      backbone='xception')

    model.load_weights("deeplabv3_v4.h5")
    model.compile(optimizer=tf.optimizers.Adam(1e-4), loss=iou_loss, metrics=[iou_metric, sym_dif])


# ---------------------------------------------------------------
#                       Image Processing
# ---------------------------------------------------------------
def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255

    # return the processed image
    return image

# ---------------------------------------------------------------
#                      Image visualisation
# ---------------------------------------------------------------
def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    fig = plt.figure(figsize=(15, 5))
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 6])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = (seg_map * 255).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.2)
    plt.axis('off')
    plt.title('segmentation overlay')

    plt.grid('off')
    # plt.show()

    # here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image


def vis_segmentation2():
    """Visualizes input image, segmentation map and overlay view."""
    fig = plt.figure(figsize=(15, 5))
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 6])

    plt.subplot(grid_spec[0])
    plt.imshow(I)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = (p * 255).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(I)
    plt.imshow(seg_image, alpha=0.2)
    plt.axis('off')
    plt.title('segmentation overlay')

    plt.grid('off')
    # plt.show()

    # here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image


# ---------------------------------------------------------------
#                              App
# ---------------------------------------------------------------
@app.route('/', methods=["POST", "GET"])
def get():
    return {'hello': 'world'}


@app.route('/predict', methods=["POST", "GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    global I, p

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(256, 256))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            preds = preds.reshape(preds.shape[:3])
            data["predictions"] = []

            # loop over the results a
            # nd add them to the list of
            # returned predictions
            data["predictions"].append(preds)

            # Visualize the mask
            im = vis_segmentation(image[0], preds[0])

            # indicate that the request was a success
            data["success"] = True

            I = image[0]
            p = preds[0]

            with open('Drop_images/temp.png', 'wb') as handle:
                response = requests.get('http://localhost:5000/plot', stream=True)

                if not response.ok:
                    print(response)

                for block in response.iter_content(1024):
                    if not block:
                        break

                    handle.write(block)

            # return the data dictionary as a JSON response
            return flask.send_file(im,
                                   attachment_filename='plot.png',
                                   mimetype='image/png')

        if flask.request.form.get('image'):
            # read the image in PIL format
            image = Image.open('./Drop_images/' + flask.request.form["image"])

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(256, 256))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            preds = preds.reshape(preds.shape[:3])
            data["predictions"] = []

            # preds = np.zeros((224, 224))

            # loop over the results and add them to the list of
            # returned predictions
            data["predictions"].append(preds)

            # Visualize the mask
            im = vis_segmentation(image[0], preds[0])

            # indicate that the request was a success
            data["success"] = True

            I = image[0]
            p = preds[0]

            with open('Drop_images/temp.png', 'wb') as handle:
                response = requests.get('http://localhost:5000/plot', stream=True)

                if not response.ok:
                    print(response)

                for block in response.iter_content(1024):
                    if not block:
                        break

                    handle.write(block)

            # return the data dictionary as a JSON response
            # return  flask.jsonify(data)
            return flask.send_file(im,
                                   attachment_filename='plot.png',
                                   mimetype='image/png')

    else:
        return '''<title>CPSS - Predict Section</title>
                      <h1>Car Park Semantic Segmentation</h1>
                      <h2>Predict Section</h2>
                      <form method="post">
                      <p> </p>
                      Image Name: <input type="text" name="image"><br>
                      <input type="submit" value="Submit" style="height:50px; width:50px"><br>
                      </form>'''


@app.route('/predict_wkt', methods=["POST", "GET"])
def predict_wk():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(256, 256))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            preds = preds.reshape(preds.shape[:3])
            poly = U.Mask2Poly(preds)

            res = {'images': flask.request.form["image"],
                   'poly_WKT': poly}
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            data["predictions"].append(res)

            # indicate that the request was a success
            data["success"] = True

            with open('Drop_images/predictions.txt', 'w') as outfile:
                json.dump(data, outfile)

            # return the data dictionary as a JSON response
            return flask.jsonify(data)

        if flask.request.form.get('image'):
            # read the image in PIL format
            image = Image.open('./Drop_images/' + flask.request.form["image"])

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(256, 256))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            preds = preds.reshape(preds.shape[:3])
            poly = U.Mask2Poly(preds)

            res = {'images': flask.request.form["image"],
                   'poly_WKT': poly}
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            data["predictions"].append(res)

            # indicate that the request was a success
            data["success"] = True


            with open('Drop_images/predictions.txt', 'w') as outfile:
                json.dump(data, outfile)

            # return the data dictionary as a JSON response
            return flask.jsonify(data)

    else:
        return '''<title>CPSS - Predict Section</title>
                      <h1>Car Park Semantic Segmentation</h1>
                      <h2>Predict WKT Section</h2>
                      <form method="post">
                      <p> </p>
                      Image Name: <input type="text" name="image"><br>
                      <input type="submit" value="Submit" style="height:50px; width:50px"><br>
                      </form>'''


@app.route('/predict/batches', methods=["POST", "GET"])
def predict_batches():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("path"):
            # read the image in PIL format
            path = flask.request.files["path"].read()
            files_name = os.listdir(path)
            for file in files_name:
                if len(file.split(sep='.'))>1:
                    if file.split(sep='.')[1] not in ['png', 'jpg', 'tiff']:
                        files_name.remove(file)
                else:
                    files_name.remove(file)

            data["predictions"] = []

            for file in files_name:
                image = Image.open(path + '/' + file)

                # preprocess the image and prepare it for classification
                image = prepare_image(image, target=(256, 256))

                # classify the input image and then initialize the list
                # of predictions to return to the client
                preds = model.predict(image)
                preds = preds.reshape(preds.shape[:3])
                poly = U.Mask2Poly(preds)

                res = {'images': file,
                        'poly_WKT': poly}

                # loop over the results and add them to the list of
                # returned predictions
                data["predictions"].append(res)

            # indicate that the request was a success
            data["success"] = True


            with open('Drop_images/predictions.txt', 'w') as outfile:
                json.dump(data, outfile)


            # return the data dictionary as a JSON response
            return flask.jsonify(data)

        if flask.request.form.get('path'):
            path = flask.request.form["path"]
            files_name = os.listdir(path)
            for file in files_name:
                if file.split(sep='.')[1] not in ['png', 'jpg', 'tiff']:
                    files_name.remove(file)

            data["predictions"] = []

            for file in files_name:
                image = Image.open(path + '/' + file)

                # preprocess the image and prepare it for classification
                image = prepare_image(image, target=(256, 256))

                # classify the input image and then initialize the list
                # of predictions to return to the client
                preds = model.predict(image)
                preds = preds.reshape(preds.shape[:3])
                poly = U.Mask2Poly(preds)

                res = {'images': file,
                       'poly_WKT': poly}

                # loop over the results and add them to the list of
                # returned predictions
                data["predictions"].append(res)

            # indicate that the request was a success
            data["success"] = True


            with open('Drop_images/predictions.txt', 'w') as outfile:
                json.dump(data, outfile)


            # return the data dictionary as a JSON response
            return flask.jsonify(data)

    else:
        return '''<title>CPSS - Predict Section</title>
                      <h1>Car Park Semantic Segmentation</h1>
                      <h2>Predict WKT Section</h2>
                      <form method="post">
                      <p> </p>
                      Folder Path: <input type="text" name="path"><br>
                      <input type="submit" value="Submit" style="height:50px; width:50px"><br>
                      </form>'''


@app.route('/plot', methods=["POST", "GET"])
def plot_result():
    bytes_obj = vis_segmentation2()

    return flask.send_file(bytes_obj,
                           attachment_filename='plot.png',
                           mimetype='image/png')



if __name__ == '__main__':
    print(("...Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(debug=True, host='0.0.0.0')
