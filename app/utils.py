import struct
import redis
import cv2
import pybase64 as base64
import numpy as np


def image_encode(image):
    # image -> bytes -> string
    input_bytes = cv2.imencode(".jpg", image)[1].tostring()
    input_image = base64.b64encode(input_bytes)
    image_content = input_image.decode("utf-8")
    return image_content


def image_decode(image_str):
    image_bytes = base64.b64decode(image_str)
    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def redis_encode(r, embedding, name, time):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    # embedding=np.array(embedding)
    h, w = embedding.shape
    shape = struct.pack('>II', h, w)
    encoded_embedding = shape + embedding.tobytes()

    # Store encoded data in Redis
    r.set(name, encoded_embedding)
    r.expire(name, time)
    return



# get user embedding from redis
def redis_decode(r, name):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded_embedding = r.get(name)
    h, w = struct.unpack('>II', encoded_embedding[:8])
    decoded_embedding = np.frombuffer(encoded_embedding, dtype=np.float32, offset=8).reshape(h, w)

    return decoded_embedding


# get user info(name, embedding, left time) from redis
def get_user_info():
    r = redis.Redis(host='localhost', port=6379, db=0)
    name_list, embedding_list, time_list = [], [], []
    keys = r.keys()

    if not keys:
        return name_list, embedding_list, time_list

    for name in keys:
        name_list.append(name)
        embedding_list.append(redis_decode(r, name).squeeze())
        time_list.append(r.ttl(name))

    embedding_list = np.array(embedding_list)

    return name_list, embedding_list, time_list


# matching n target embeddings with m database embeddings
def compute_with_db_quick(embeddings, th=0.3):
    name_list, embedding_list, time_list = get_user_info()
    if len(embedding_list) == 0:  # if no embedding in database
        names = ['unknown' for idx in range(len(embeddings))]
        times = [-2 for idx in range(len(embeddings))]
        scores = np.zeros((1, len(embeddings)))
        return names, scores, times
    similarity = np.dot(embeddings, embedding_list.T)  # get sim_matrix (n*m)
    scores = np.max(similarity, axis=1)  # highest sim as score for each target (n)

    idxes = np.argmax(similarity, axis=1)
    idxes[scores < th] = len(name_list)
    name_list.append('unknown')  # score<th: name = "unknown"
    time_list.append(-2)  # score<th: time = -2
    names = [name_list[idx] for idx in idxes]
    times = [time_list[idx] for idx in idxes]

    return names, scores, times


def check(name):
    r = redis.Redis(host='localhost', port=6379, db=0)
    if (r.exists(name)):
        status = 0
    else:
        status = 1
    return status
