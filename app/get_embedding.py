import os
os.chdir("..")
# import face_model
import argparse
import cv2
import sys
import numpy as np
import pandas as pd
import traceback

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-dir', default="/mnt/hdd1/wkc/FR_demo/FR", help='')
parser.add_argument('--image-size', default='112,112', help='')
# parser.add_argument('--model', default='model_50/model-0000, 0', help='path to load model.')
parser.add_argument('--model', default='/mnt/hdd1/clliao/SOD_FR/deploy_models/model_alignt_person/model, 792', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parsers = parser.parse_args()


# def get_images(input_images):
#     """
#     input: (1) path of directory of many people's face image directory named by face owners.
#            (2) path of directory of face images belong to one person named by the face owner.
#            (3) path of one image file
#     return: dictionary with person name as "key" and list of paths of images as "value"
#     """
#     # print "image source: %s" % input_images
#     face_dict = dict()
#     # input_images is a directory contain many face images or many directorys with face image in it.
#     if os.path.isdir(input_images):
#         element_list = os.listdir(input_images)
#         # print "contain:"
#         # print element_list
#         # element may be image file or a directory of images named by the face owner of images
#         for element in element_list:
#             element_path = os.path.join(input_images, element)
#             # directory: element is person name
#             if os.path.isdir(element_path):
#                 face_dict[element] = dict()
#                 file_list = os.listdir(element_path)
#                 for filename in file_list:
#                     file_path = os.path.join(element_path, filename)
#                     face_dict[element][file_path] = -1
#             # image file: element is image file name
#             else:
#                 person_name = input_images.split('/')[-1]
#                 if person_name not in face_dict:
#                     face_dict[person_name] = dict()
#                 face_dict[person_name][os.path.join(input_images, element)] = -1
#     # input_images is one image file
#     else:
#         face_dict['unknown'] = dict()
#         face_dict['unknown'][input_images] = -1
#
#     return face_dict


if __name__ == "__main__":
    args = parser.parse_args()
    # model = face_model.FaceModel(args)

    # version 1 (lfw version). several picture for one person (without db supporting)
    """
    face_dict = get_images(os.path.abspath(args.image_dir))
    for person in face_dict.keys():
        for raw_img_path in face_dict[person].keys():
            raw_img = cv2.imread(raw_img_path)
            aligned_img = model.get_input(raw_img)
            face_dict[person][raw_img_path] = model.get_feature(aligned_img)
    print face_dict
    """

    # version 2 (wkc version). one person with one picture
    """
    face_dict = {
        "user": list(),
        "vector": list()
    }
    images_list = os.listdir(args.image_dir)
    for image_file in images_list:
        raw_img_path = os.path.join(args.image_dir, image_file)
        if os.path.isfile(raw_img_path):
            raw_img = cv2.imread(raw_img_path)
            try:
                aligned_img = model.get_input(raw_img)
                face_dict["user"].append(image_file.split('.')[0])
                face_dict["vector"].append(model.get_feature(aligned_img))
            except:
                # traceback.print_exc(file=sys.stdout)
                print "broken image(error code: %s): %s" % (str(aligned_img), raw_img_path)
                pass
    # print face_dict
    """

    # # save npy dictionary
    # np.save("../db.npy", face_dict)
    # print "npy db saved"

    # load npy db
    db_dir = os.path.abspath(os.path.join(os.curdir, "db"))
    print db_dir
    db_from_npy = np.load(os.path.join(db_dir, "db_full.npy")).item()  # py2
    # db_from_npy = np.load(os.path.join(db_dir, "db_full.npy"), encoding='latin1').item()  # py3
    # print db_from_npy.keys()

    # change format for pandas db
    face_dict = {
        "user": list(),
        "vector": list()
    }
    for id in db_from_npy.keys():
        face_dict["user"].append(id)
        face_dict["vector"].append(db_from_npy[id])

    # convert dictionary to pandas
    pandas_db = pd.DataFrame.from_dict(face_dict)
    # print(pandas_db)

    # save pandas dictionary
    # pandas_db.to_hdf('../db_py2.h5', 'table')
    # print("hdf5 db saved")

    # load pandas db
    pandas_db = pd.read_hdf(os.path.join(db_dir, 'db_py2.h5'), 'table')
    # print(pandas_db)
    pandas_series_db = pandas_db.to_dict("series")
    db = dict(zip(pandas_series_db["user"], pandas_series_db["vector"]))
    print(db.keys())

    # ==============================================
    # get feature all vectors of the specified person
    # face_dict[<person_name>].values()

    # compute similarity of two feature
    # np.dot(f1, f2.T)

    # compute similarity of new feature vector and the specified person
    # np.dot(nf, np.asarray(face_dict[<person_name>].values()))

