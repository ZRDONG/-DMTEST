# -*- coding: utf-8 -*-

"""
description: MySQL Env Utils
"""

import sys
import time
# import json
import server as server
# import httplib
import dmPython
import requests
# import xmlrpclib

from configs import instance_config
# from warnings import filterwarnings
#
# filterwarnings('error', category=MySQLdb.Warning)

#'dm_global_status_tps', 'dm_global_status_qps', 'dm_global_status_ips', 'dm_global_status_nioips',
#                       'dm_global_status_nio_ops', 'dm_global_status_fio_ips', 'dm_global_status_fio_ops',
#                       'dm_global_status_mem_used', 'dm_global_status_sessions', ' dm_global_status_active_sessions',
#                       'dm_global_status_task_waiting', 'dm_global_status_task_ready',
#                       'dm_global_status_task_total_wait_time', 'dm_global_status_avg_wait_time',
#                       'dm_global_status_threads'
value_type_metrics = []


def time_start():
    return time.time()


def time_end(start):
    end = time.time()
    delay = end - start
    return delay


def get_metric_type(metric):
    if metric in value_type_metrics:
        return 'value'
    else:
        return 'counter'


def get_metrics(db_config):
    # 连接到 MySQL 数据库
    conn = dmPython.connect(
        db_config['user'],
        db_config['passwd'],
        db_config['host:port']
    )

    cursor = conn.cursor()

    # 定义查询字典
    queries = {
        'dm_global_status_tps': '''select stat_val from sys.v$sysstat where name = 'transaction total count';''',
        'dm_global_status_qps': '''select stat_val from sys.v$sysstat where name = 'select statements';''',
        'dm_global_status_ips': '''select stat_val from sys.v$sysstat where name = 'insert statements';''',
        'dm_global_status_nioips': '''select stat_val from sys.v$sysstat where name = 'total bytes received from client';''',
        'dm_global_status_nio_ops': '''select stat_val from sys.v$sysstat where name = 'total bytes sent to client';''',
        'dm_global_status_fio_ips': '''select stat_val * page from sys.v$sysstat where name = 'physical read count';''',
        'dm_global_status_fio_ops': '''select stat_val * page from sys.v$sysstat where name = 'physical write count';''',
        'dm_global_status_mem_used': '''select stat_val from sys.v$sysstat where name = 'memory used bytes';''',
        'dm_global_status_cpu_use_rate': '''select stat_val from sys.v$sysstat where name = 'os DM database cpu rate';''',
        'dm_global_status_sessions': '''select count(1) from v$sessions;''',
        'dm_global_status_active_sessions': '''select count(1) from v$sessions where state = 'ACTIVE';''',
        'dm_global_status_task_waiting': '''select waiting from v$task_queue;''',
        'dm_global_status_task_ready': '''select ready from v$task_queue;''',
        'dm_global_status_task_total_wait_time': '''select total_wait_time from v$task_queue;''',
        'dm_global_status_avg_wait_time': '''select average_wait_time from v$task_queue;''',
        'dm_global_status_threads': '''select count(1) from sys.v$threads;'''
    }

    # 初始化结果字典
    results = {}

    # 执行每个查询并存入结果字典
    for key, query in queries.items():
        cursor.execute(query)
        result = cursor.fetchone()  # 获取第一个查询结果的元组
        if result is not None:
            # 提取元组中的值，并将其转换为数字
            value = int(result[0])
            results[key] = value
        else:
            # 如果查询结果为空，将结果设为 None 或其他适当的值
            results[key] = None

    # 打印结果字典
    for key, value in results.items():
        print(f"{key}: {value}")

    # 关闭游标和连接
    cursor.close()
    conn.close()

    return results


# class TimeoutTransport(xmlrpclib.Transport):
#     timeout = 30.0
#
#     def set_timeout(self, timeout):
#         self.timeout = timeout
#
#     def make_connection(self, host):
#         h = httplib.HTTPConnection(host, timeout=self.timeout)
#         return h


def get_state():
    """ get mysql state
    Args:
        server_ip: str, ip address
    """

    # # 创建一个 TimeoutTransport 实例并设置超时时间为 60 秒
    # transport = TimeoutTransport()
    # transport.set_timeout(60)
    #
    # # 创建 XML-RPC 服务器代理对象，连接到指定的服务器 IP 地址
    # s = xmlrpclib.ServerProxy('http://%s:20000' % server_ip, transport=transport)

    try:
        # 调用 XML-RPC 服务器上的 get_state 方法，获取 MySQL 实例的状态
        m = server.get_state()
    except Exception:
        return True

    # 如果结果为-1  证明数据库也不处于正常运行状态
    if m == -1:
        sys.stdout.write('.')
        sys.stdout.flush()
        return False

    return True


def restart_database(instance_name, configuration):
    """ Modify the configurations by restarting the mysql through Docker通过重新启动 Docker 中的 MySQL 修改配置
    Args:
        server_ip: str, instance's server IP Addr     server_ip (str): 实例所在服务器的 IP 地址
        instance_name: str, instance's name      instance_name (str): 实例名称
        configuration: dict, configurations      configuration (dict): 配置参数字典
    """
    # # 创建一个 TimeoutTransport 实例并设置超时时间为 60 秒
    # transport = TimeoutTransport()
    # transport.set_timeout(60)

    # # 创建 XML-RPC 服务器代理对象，连接到指定的服务器 IP 地址
    # # XML-RPC 服务器代理对象是一个用于与 XML-RPC 服务器进行通信的客户端工具
    # s = xmlrpclib.ServerProxy('http://%s:20000' % server_ip, transport=transport)
    params = []
    # 构建knobs参数列表，格式为 "key:value"，并以逗号分隔
    for k, v in configuration.items():
        params.append('%s:%s' % (k, v))
    params = ','.join(params)

    # 循环启动mysqlserver
    while True:
        try:
            server.start_dm(instance_name, params)
        except Exception as e:
            print(f"出现异常: {e}")
            time.sleep(5)
        break

    return True


def test_dameng(instance_name):
    """ Test the mysql instance to see whether if it has been restarted
    Args
        instance_name: str, instance's name
    """

    db_config = instance_config[instance_name]
    try:
        db = dmPython.connect(
            db_config['user'],
            db_config['passwd'],
            db_config['host:port']
        )
    except Exception:
        return False
    db.close()
    return True


# def get_tencent_instance_info(instance_name):
#     """ get Tencent Instance information
#     Args:
#         url: str, request url
#         instance_name: str, instance_name
#     Return:
#         info: tuple, (mem, disk)
#     Raises:
#         Exception: setup failed
#     """
#     db_info = instance_config[instance_name]
#     instance_id = db_info['instance_id']
#     operator = db_info['operator']
#     url = db_info['server_url']
#     data = dict()
#     data["instanceid"] = instance_id
#     data["operator"] = operator
#     para_list = []
#
#     data["para_list"] = para_list
#     data = json.dumps(data)
#     data = "data=" + data
#     r = requests.get(url + '/get_inst_info.cgi', data)
#     response = json.loads(r.text)
#     print data
#     print(response)
#     # default 32GB
#     mem = int(response.get('mem', 12*1024))*1024*1024
#     # default 100GB
#     disk = int(response.get('disk', 100))*1024*1024*1024
#     return mem, disk


def read_machine():
    """ Get the machine information, such as memory and disk

    Return:

    """
    # 打开 /proc/meminfo 文件，读取内存信息
    f = open("/proc/meminfo", 'r')
    # 读取文件的第一行
    line = f.readlines()[0]
    f.close()
    # 去除行尾的换行符
    line = line.strip('\r\n')
    # 获取内存总量，以字节为单位
    total = int(line.split(':')[1].split()[0]) * 1024
    return total
