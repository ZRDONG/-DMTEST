# -*- coding: utf-8 -*-
from flask import Response, json

from application._init_ import init_app, db
from application.ProductModel import Product


app = init_app()


@app.route("/query_all")
def query_all():
    # 查询所有用户数据, 返回User列表,
    data = Product.query.all()
    response_data = []
    for item in data:
        info = dict()
        info['DBNAME'] = item.DBNAME
        info['DBTYPE'] = item.DBTYPE
        info['DBDEFALT'] = item.DBDEFALT
        info['DBRANGE'] = item.DBRANGE
        info['NOTE'] = item.NOTE
        response_data.append(info)

    return Response(json.dumps(response_data), mimetype='application/json')

#
# @app.route("/query_first")
# def query_filter():
#     # 查询所有用户数据, 返回User列表,
#     data = Product.query.filter_by(NAME='面纱').first()
#     response_data = dict()
#     response_data['PRODUCTID'] = data.PRODUCTID
#     response_data['NAME'] = data.NAME
#     response_data['AUTHOR'] = data.AUTHOR
#     response_data['DESCRIPTION'] = data.DESCRIPTION
#
#     return Response(json.dumps(response_data), mimetype='application/json')
#
#
# @app.route("/add_one")
# def add_one():
#     # 添加一条数据
#     # 将图片转化为二进制数据进行存储
#     product = Product(PRODUCTID=1,
#                       NAME='万历十五年',
#                       AUTHOR='黄仁宇',
#                       PUBLISHER='中华书局',
#                       PUBLISHTIME='1981-09-01',
#                       PRODUCTNO='9787101046126',
#                       SATETYSTOCKLEVEL=10, ORIGINALPRICE=39.8000, NOWPRICE=20.0000,
#                       DISCOUNT=5.0,
#                       DESCRIPTION='《万历十五年》以1587年为关节点，在历史的脉络中延伸，从政治、经济、军事等各个方面的历史大事与人物着手，记叙了明朝中晚期的种种社会矛盾和开始走向衰败的迹象。',
#                       PHOTO=open('C:/Users/Administrator/PycharmProjects/FlaskDm/application/OIP.jpg', 'rb').read(),
#                       TYPE='16', PAPERTOTAL=684, WORDTOTAL=68000,
#                       SELLSTARTTIME='2006-03-20', SELLENDTIME='2023-11-1'
#                       )
#     db.session.add(product)
#     db.session.commit()
#     return Response('数据插入成功，本次插入数据的PRODUCTID为{}'.format(product.PRODUCTID), mimetype='application/json')
#
#
# @app.route("/add_list")
# def add_list():
#     # 一次添加多条数据
#     data_list = [
#         Product(PRODUCTID=2,
#                 NAME='面纱', AUTHOR='威廉·萨默塞特·毛姆', PUBLISHER='人民文学出版社', PUBLISHTIME='2020-04-01',
#                 PRODUCTNO='9787101046127',
#                 SATETYSTOCKLEVEL=10, ORIGINALPRICE=39.8000, NOWPRICE=20.0000,
#                 DISCOUNT=5.0,
#                 DESCRIPTION='小说的故事发生在香港和一个叫“湄潭府”的地方。女主人公凯蒂·费恩因为和香港助理布政司查理通奸，被丈夫瓦尔特（细菌学家）发现后胁迫她去了霍乱横行的湄潭府，最终瓦尔特不幸染病死去，凯蒂回到香港，重投查理怀抱后羞愧不已，最终回到英国和父亲和解，并和父亲同往巴哈马群岛生活。',
#                 PHOTO=open('C:/Users/Administrator/PycharmProjects/FlaskDm/application/ORF.jpg', 'rb').read(),
#                 TYPE='16', PAPERTOTAL=684, WORDTOTAL=68000, SELLSTARTTIME='2006-03-20',
#                 SELLENDTIME='2023-11-1'),
#         Product(PRODUCTID=3,
#                 NAME='乖，摸摸头', AUTHOR='大冰', PUBLISHER='湖南文艺出版社', PUBLISHTIME='2014-10-01',
#                 PRODUCTNO='9787101046129',
#                 SATETYSTOCKLEVEL=10, ORIGINALPRICE=39.8000, NOWPRICE=20.0000,
#                 DISCOUNT=5.0,
#                 PHOTO=open('C:/Users/Administrator/PycharmProjects/FlaskDm/application/OWW.jpg', 'rb').read(),
#                 DESCRIPTION='此书记录了大冰十余年的江湖游历，以及他和他朋友们的爱与温暖的传奇故事。',
#                 TYPE='16', PAPERTOTAL=684, WORDTOTAL=68000, SELLSTARTTIME='2006-03-20',
#                 SELLENDTIME='2023-11-1')
#     ]
#
#     db.session.add_all(data_list)
#     db.session.commit()
#     return Response('数据插入成功，本次共插入{}条数据'.format(len(data_list)))
#
#
# @app.route("/updata_info")
# def updata_info():
#     # 直接根据条件修改
#     Product.query.filter(Product.PRODUCTID == 2).update({Product.NAME: '面纱(中文)'})
#     db.session.commit()
#
#     # 查询修改后的值
#     data = Product.query.filter_by(PRODUCTID=2).first()
#     response_data = dict()
#     response_data['PRODUCTID'] = data.PRODUCTID
#     response_data['NAME'] = data.NAME
#
#     return Response(json.dumps(response_data), mimetype='application/json')
#
#
# @app.route("/delete_info")
# def delete_info():
#     # 直接根据条件修改
#     Product.query.filter(Product.PRODUCTID == 1).delete()
#     db.session.commit()
#
#     # 查询修改后的值
#     t_count = Product.query.count()
#
#     return Response('数据删除成功！目前表中还有{}条数据！'.format(t_count))

@app.route("/")
def HELLO_DM():
    return 'HELLO DM'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
