from flask import Flask
import os

from config import config, Config


def create_app(config_name):
    app = Flask(__name__, instance_relative_config=True)

    # config
    app.config.from_object(config[config_name])
    app.config.from_pyfile('config.py')

    #from app import views


    # # initalize
    # @app.before_first_request
    # def setup():
    #     mongo.db.records.create_index([('time_stamp', -1)], background=True)
    #     if not os.path.exists('./data'):
    #         os.mkdir('data')
    #     if not os.path.exists('./instance'):
    #         os.mkdir('instance')
    #     if not os.path.exists('./records'):
    #         os.mkdir('records')

    return app

