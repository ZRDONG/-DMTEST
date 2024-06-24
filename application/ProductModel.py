# -*- coding: utf-8 -*-


# 构建模型类
from application._init_ import db


# 构建模型类
class Product(db.Model):
    # 设置表名，表名默认为类名小写
    __tablename__ = 'DBPARAMETER'
    DBNAME = db.Column(db.String(50), primary_key=True)
    DBTYPE = db.Column(db.String(50))
    DBDEFALT = db.Column(db.String(50))
    DBRANGE = db.Column(db.String(50))
    NOTE = db.Column(db.String(200))

    # PRODUCTNO = db.Column(db.String(25))
    # SATETYSTOCKLEVEL = db.Column(db.Integer)
    # ORIGINALPRICE = db.Column(db.Numeric(19, 4))
    # NOWPRICE = db.Column(db.Numeric(19, 4))
    # DISCOUNT = db.Column(db.Numeric(2, 1))
    # DESCRIPTION = db.Column(db.Text)
    # # 图片转化为二进制数据进行存储
    # PHOTO = db.Column(db.LargeBinary)
    # TYPE = db.Column(db.String(5))
    # PAPERTOTAL = db.Column(db.Integer)
    # WORDTOTAL = db.Column(db.Integer)
    # SELLSTARTTIME = db.Column(db.Date)
    # SELLENDTIME = db.Column(db.Date)

    def __repr__(self):  # 自定义 交互模式 & print() 的对象打印
        return "(%S, %s, %s, %s, %s)" % (self.DBNAME, self.DBTYPE, self.DBDEFALT, self.DBRANGE, self.NOTE)
