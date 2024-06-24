# # from sqlalchemy import create_engine
# # from sqlalchemy.exc import SQLAlchemyError
# import dmPython
# # # 替换为你的数据库连接字符串
# # DATABASE_URL = 'dm+dmPython://SYSDBA:SYSDBA@0.0.0.0:5236'
# conn = dmPython.connect(
#     'SYSDBA',
#     'SYSDBA',
#     'localhost:5236'
# )
# cursor = conn.cursor()
#
# # 定义查询字典
# queries = {
#     'dbname': '''select DBNAME from PARAMETER.DBPARAMETER;''',
#     # 'dm_global_status_qps': '''select stat_val from sys.v$sysstat where name = 'select statements';''',
#     # 'dm_global_status_ips': '''select stat_val from sys.v$sysstat where name = 'insert statements';''',
#     # 'dm_global_status_nioips': '''select stat_val from sys.v$sysstat where name = 'total bytes received from client';''',
#     # 'dm_global_status_nio_ops': '''select stat_val from sys.v$sysstat where name = 'total bytes sent to client';''',
#     # 'dm_global_status_fio_ips': '''select stat_val * page from sys.v$sysstat where name = 'physical read count';''',
#     # 'dm_global_status_fio_ops': '''select stat_val * page from sys.v$sysstat where name = 'physical write count';''',
#     # 'dm_global_status_mem_used': '''select stat_val from sys.v$sysstat where name = 'memory used bytes';''',
#     # 'dm_global_status_cpu_use_rate': '''select stat_val from sys.v$sysstat where name = 'os DM database cpu rate';''',
#     # 'dm_global_status_sessions': '''select count(1) from v$sessions;''',
#     # 'dm_global_status_active_sessions': '''select count(1) from v$sessions where state = 'ACTIVE';''',
#     # 'dm_global_status_task_waiting': '''select waiting from v$task_queue;''',
#     # 'dm_global_status_task_ready': '''select ready from v$task_queue;''',
#     # 'dm_global_status_task_total_wait_time': '''select total_wait_time from v$task_queue;''',
#     # 'dm_global_status_avg_wait_time': '''select average_wait_time from v$task_queue;''',
#     # 'dm_global_status_threads': '''select count(1) from sys.v$threads;'''
# }
#
#
# # 执行每个查询并存入结果字典
# cursor.execute('select DBNAME from PARAMETER.DBPARAMETER;')
# result = cursor.fetchall()  # 获取第一个查询结果的元组
# print(result)
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# 替换为你的数据库连接字符串
DATABASE_URL = 'dm+dmPython://SYSDBA:SYSDBA@localhost:5236'

try:
    # 创建数据库连接
    engine = create_engine(DATABASE_URL)

    # 连接到数据库
    with engine.connect() as connection:
        # 执行一个简单的查询
        result = connection.execute('''SELECT "PARAMETER.DBPARAMETER"."DBNAME" AS "PARAMETER.DBPARAMETER_DBNAME", "PARAMETER.DBPARAMETER"."DBTYPE" AS "PARAMETER.DBPARAMETER_DBTYPE", "PARAMETER.DBPARAMETER"."DBDEFALT" AS "PARAMETER.DBPARAMETER_DBDEFALT", "PARAMETER.DBPARAMETER"."DBRANGE" AS "PARAMETER.DBPARAMETER_DBRANGE", "PARAMETER.DBPARAMETER"."NOTE" AS "PARAMETER.DBPARAMETER_NOTE" 
FROM PARAMETER.DBPARAMETER''')
        # 获取结果
        for row in result:
            print("连接成功:", row)
except SQLAlchemyError as e:
    print("连接失败:", str(e))
