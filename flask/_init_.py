# -*- coding: utf-8 -*-

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_app():
    app = Flask(__name__)

    # 解决页面中问乱码问题
    app.config['JSON_AS_ASCII'] = False
    # 设置数据库链接地址
    app.config['SQLALCHEMY_DATABASE_URI'] = "dm+dmPython://PARAMETER:sysdba@127.0.0.1:5236"
    # 设置显示底层执行的sql语句
    app.config['SQLALCHEMY_ECHO'] = True
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

    # 初始化组件对象，关联flask应用
    db.init_app(app)

    return app
