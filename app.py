from threading import Lock, Event
from flask_socketio import SocketIO, emit
import os
from PIL import Image, ImageFilter
from flask import Flask, request, Response, jsonify
from flask import current_app as app
from time import sleep
import pybase64 as base64
import cv2
import numpy as np
import requests
import json
import io
#from app import views
from app.utils import image_encode, image_decode, redis_decode, get_user_info, compute_with_db_quick, redis_encode, check
from app import face_model

import argparse
import struct
import redis

app = Flask(__name__)
socketio = SocketIO(app)

# thread_lock = Lock()
# for CORS, make it work for more than localhost
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response



@app.route('/')
def index():
    return Response('Tensor Flow object detection')


@app.route('/local')
def local():
    return Response(open('./static/index.html').read(), mimetype="text/html")


# @app.route('/fd', methods=['GET', 'POST'])
# def fd():
#     try:
#         # req_data = request.get_json()
#         req_data = fd_req
#         if req_data['get_align']==True:
#             ## FD function body
#             pass
#             return jsonify({'cord': [1, 2, 3, 4]}), 201  #
#
#         # call FD alignment function
#         return jsonify({"aligned": ["QAFGEYTHY45AFHYH4536y5"], "cord": [1, 2, 3, 4]}), 201     #
#
#     except Exception as e:
#         print('POST /fd error: %e' % e)
#         return e

@app.route('/register', methods=['POST'])
def register():
    client_req = request.json
    img = request.json['crop_face']
    headers = {'content-type': 'application/json; charset=utf-8'}

    if (check(request.json['id']) == 0):
        return Response(json.dumps({"status": 3, "msg": "ID exists!"}), mimetype='application/json')

    '''fd_input = {'cam': img[0], 'get_box': 'False'}
    fd_res = requests.post("http://172.16.120.54:6666/fd", data=json.dumps(fd_input), headers=headers)
    fd_result = fd_res.json()

    if (fd_result['faces']==[]):
        return Response(json.dumps({"status": 0, "msg": "no face!"}), mimetype='application/json')
    elif (len(fd_result['faces'])>1):
        return Response(json.dumps({"status": 1, "msg": "muti face!"}), mimetype='application/json')

    matching_input = {'crop_face_str': fd_result['faces']}  ###fd output
    '''
    matching_input = {'crop_face_str': img}
    mat_res = requests.post("http://127.0.0.1:5000/fr/matching", data=json.dumps(matching_input), headers=headers)
    mat_result = mat_res.json()

    if (mat_result['names'] != [u'unknown']):
        return Response(json.dumps({"status": 2, "msg": "embedings  exists!"}), mimetype='application/json')
    else:
        print('hi')
        r = redis.Redis(host='localhost', port=6379, db=0)
        embedding = mat_result['embeddings'][0]
        embedding = np.array(embedding).astype(np.float32)
        embedding = np.expand_dims(embedding, axis=0)
        redis_encode(r, embedding, client_req['id'], 300)
        return Response(json.dumps({"status": 4, "msg": "success!"}), mimetype='application/json')


# @app.route('/fr/front', methods=['GET', 'POST'])
# def front():
#     try:
#         # req_data = request.get_json()
#         req_data = front_req
#         keys = ['cam', 'th_fd', 'th_mat']
#         if set(keys) == set(req_data.keys()):
#             tmp = []
#             #  fd, fr/matching Body
#             response = ['id', [1, 2, 3, 4], 180]
#             return jsonify(response), 201
#
#     except Exception as e:
#         print('POST /front error: %e' % e)
#         return e

@app.route('/fr/matching', methods=['POST'])
def matching():
    req_data = request.get_json()
    faces = req_data['crop_face_str']

    embeddings = []
    for face in faces:
        face = image_decode(face)

        # ## tmp
        # aligned_img, bbox = fmodel.get_input(face)
        # embedding = fmodel.get_feature(aligned_img[0].transpose(2, 0, 1))
        # ##

        embedding = fmodel.get_feature(cv2.resize(face, (112, 112)).transpose(2, 0, 1))
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    if 'th_mat' in req_data:
        names, scores, times = compute_with_db_quick(embeddings, th=float(req_data['th_mat']))
    else:
        names, scores, times = compute_with_db_quick(embeddings)

    return jsonify({'names': names, 'scores': scores.tolist(), 'times': times, 'embeddings': embeddings.tolist()})



### Socket IO

@socketio.on('start')
def socket_message(message):
    print('start: ', message)

@socketio.on('log')
def socket_log(message):
    print(message)

@socketio.on('catch-frame')
def catch_frame(data):
    # print(data)
    try:
        image_file = data['image']  # get the image
        # print(type(image_file))    #<class 'bytes'>
        threshold = data['threshold']
        if threshold is None:
            threshold = 0.5
        else:
            threshold = float(threshold)

        # PIL bytes to Image datatype
        image_object = Image.open(io.BytesIO(image_file))
        print('--------------------------------')
        # finally run the image through FR`
        image_object = cv2.cvtColor(np.asarray(image_object), cv2.COLOR_RGB2BGR)
        image_object = image_encode(image_object)
        body = {"image": image_object}
        URL = "http://127.0.0.1:5555/"
        print('123')

        objects = requests.post(URL, data=json.dumps(body)).text
        # objects = object_detection_api.get_objects(image_object, threshold)
        emit('return-frame', objects)
        # print('ccccccccccccc')

    except Exception as e:
        # print('POST /image error: %e' % e)
        print(e)
        return e

    ## getting the data frames in b64 format

    ## do FD & FR (matching) process


    ## Apply to FR model and send it back to client
    ## {["tony", x, y, h, w, remain_time]...}
    # emit('response_back', data)


# @socketio.on('disconnect')
# def test_disconnect():
#     print('Client disconnected')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    # parser.add_argument('--model', default='model_50/model-0000, 0', help='path to load model.')
    parser.add_argument('--model', default='./deploy_models/model_alignt_person/model,792', help='path to load model.')
    parser.add_argument('--mtcnn_model', default='./deploy_models/mtcnn-model/', help='path to load model.')
    parser.add_argument('--retinaface', default=True, type=bool,
                        help='true : RetinaFace, false : MTCNN')
    parser.add_argument('--retina_model', default='deploy_models/retina_model/retina')
    parser.add_argument("--retina_epoch", help="test epoch", default="1220", type=int)
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--threshold', default=0.4, type=float, help='ver dist threshold')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')

    # parser.add_argument('--db_path', default="db_full_792.npy", type=str, help='path to .npy db')
    parser.add_argument('--db_path', default="./db/db_full_1081120.npy", type=str, help='path to .npy db')
    # parser.add_argument('--db_path', default="/mnt/hdd1/SOD/SOD_FR/db/db_full.npy", type=str, help='path to .npy db')
    args = parser.parse_args()
    fmodel = face_model.FaceModel(args)
    socketio.run(app, debug=True, host='0.0.0.0')
    app.run(debug=True, host='0.0.0.0', port=5555)
    # app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))
