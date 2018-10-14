import os
import logging
from flask import Flask
from .db import db


def create_app():
    app = Flask('__name__')
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_DATABASE_URI='mysql+pymysql://{}?charset=utf8'.format(os.getenv('MYSQL_URI')),
        LOGGING_LEVEL=logging.INFO
    )

    db.init_app(app)

    from . import api
    app.register_blueprint(api.bp)

    @app.route("/")
    def hello():
        current_app.logger.info('hello: {} request received from: {}'.format(
            request.method, request.remote_addr))
        return "hello wejk"

    app.logger.setLevel(app.config['LOGGING_LEVEL'])    
    app.logger.info('sever start')
    return app
