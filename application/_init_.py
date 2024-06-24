# -*- coding: utf-8 -*-

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

db = SQLAlchemy()


def init_app():
    # 创建一个 Flask 应用实例
    app = Flask(__name__)
    # 并允许来自所有域的请求
    CORS(app)

    # 解决页面中问乱码问题
    app.config['JSON_AS_ASCII'] = False
    # 设置数据库链接地址
    app.config['SQLALCHEMY_DATABASE_URI'] = "dm+dmPython://PARAMETER:sysdba@localhost:5236"
    # 设置显示底层执行的sql语句
    app.config['SQLALCHEMY_ECHO'] = True
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

    # 初始化组件对象，关联flask应用
    db.init_app(app)

    return app
