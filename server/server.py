# -*- coding: utf-8 -*-
"""
Configure Server
"""

import os
import time
import pexpect
import platform
import subprocess
import shutil
import argparse
# import ConfigParser as CP
# from SimpleXMLRPCServer import SimpleXMLRPCServer

docker = False


# 获取 DaMeng 服务器状态
def get_state():
    check_start()  # 检查并启动 DaMeng 服务器
    a = sudo_exec('sudo /home/dmdba/dmdbms/bin/DmServiceDMTEST status', 'Czy139123450988')
    decoded_a = a.decode('utf-8')
    print(decoded_a)
    # m = os.popen('service mysql status')  # 获取 DaMeng 服务器的状态信息
    # s = m.readlines()[2]  # 提取状态信息的第三行（通常包含状态）
    # s = s.split(':')[1].replace(' ', '').split('(')[0]  # 提取状态信息
    if a.find('stop') != -1:  # 如果在错误日志中找到了 'pid ended'，则说明 MySQL 服务器已停止
        return -1
    return 1


def check_start():
    # TODO 优化配置文件选择数据库路径
    a = sudo_exec('sudo /home/dmdba/dmdbms/bin/DmServiceDMTEST status', 'Czy139123450988')  # 执行命令以检查错误日志中的信息
    a = a.decode('utf-8')
    a = a.strip('\n\r')  # 去除字符串首尾的换行符和回车符
    if a.find('dead') != -1:  # 如果在错误日志中找到了 'pid ended'，则说明 MySQL 服务器已停止
        run_directory = '/dmdata/benchmarksql-5.0/run'
        # 切换到目标目录并执行命令
        original_directory = os.getcwd()  # 保存当前工作目录
        os.chdir(run_directory)  # 切换到目标目录
        try:
            start_time = time.time()
            # 要执行的命令和参数
            command = ['sudo', '/home/dmdba/dmdbms/bin/DmServiceDMTEST', 'start']
            subprocess.run(command)
            end_time = time.time()
            # 计算并打印运行时间
            execution_time = end_time - start_time
            print(f"Check_start Execution time: {execution_time} seconds")
        finally:
            os.chdir(original_directory)  # 恢复到原来的工作目录
        # sudo_exec('sudo /home/dmdba/dmdbms/bin/DmServiceDMTEST start', 'Czy139123450988')  # 启动 MySQL 服务器


def sudo_exec(cmdline, passwd):
    osname = platform.system()  # 获取操作系统名称
    if osname == 'Linux':
        prompt = r'\[sudo\] password for %s: ' % os.environ['USER']  # 设置 sudo 提示符
    elif osname == 'Darwin':   # 如果是 macOS 系统
        prompt = 'Password:'
    else:
        assert False, osname
    child = pexpect.spawn(cmdline)  # 使用 pexpect 模块生成子进程来执行命令
    idx = child.expect([prompt, pexpect.EOF], 3000)  # 等待命令执行结果，3 秒超时
    print(child.before.decode('utf-8'))
    if idx == 0:  # 如果匹配到了 sudo 提示符
        child.sendline(passwd)   # 输入密码
        child.expect(pexpect.EOF)   # 等待命令执行完成
    return child.before  # 返回命令执行结果


# 启动 MySQL 服务器
def start_dm(instance_name, configs):
    """
    Args:
        instance_name: str, MySQL Server Instance Name eg. ["mysql1", "mysql2"]  MySQL Server 实例名称，例如 ["mysql1", "mysql2"]
        configs: str, Formatted MySQL Parameters, e.g. "name:value,name:value"  格式化的 MySQL 参数，例如 "name:value,name:value"
    """

    params = configs.split(',')  # 将参数字符串分割为参数列表

    if docker:  # 如果在 Docker 中运行 MySQL 服务器
        _params = ''
        for param in params:
            pair_ = param.split(':')
            _params += ' --%s=%s' % (pair_[0], pair_[1])
        sudo_exec('sudo docker stop %s' % instance_name, '123456')  # 停止 Docker 容器
        sudo_exec('sudo docker rm %s' % instance_name, '123456')  # 删除 Docker 容器
        time.sleep(2)
        # 构建 Docker 启动命令
        cmd = 'sudo docker run --name mysql1 -e MYSQL_ROOT_PASSWORD=12345678 ' \
              '-d -p 0.0.0.0:3365:3306 -v /data/{}/:/var/lib/mysql mysql:5.6 {}'.format(instance_name, _params)
        print(cmd)  # 打印 Docker 启动命令
        sudo_exec(cmd, '123456')  # 执行 Docker 启动命令
    else:  # 如果不在 Docker 中运行 MySQL 服务器
        data_dir = '/dmdata/data/DAMENG'
        backup_path = os.path.join(data_dir, 'dm.ini.dmbak')
        target_path = os.path.join(data_dir, 'dm.ini')

        # 检查备份文件是否存在
        if os.path.exists(backup_path):
            # 直接覆盖目标文件
            shutil.copy2(backup_path, target_path)
            print(f"Restored {backup_path} to {target_path}")
        else:
            print(f"Backup file {backup_path} not found.")

        write_cnf_file(params)  # 写入 MySQL 配置文件
        # TODO 优化  写一个配置文件选择数据库路径
        # run_directory = '/dmdata/benchmarksql-5.0/run'
        # # 切换到目标目录并执行命令
        # original_directory = os.getcwd()  # 保存当前工作目录
        # os.chdir(run_directory)  # 切换到目标目录
        try:
            start_time = time.time()
            # 要执行的命令和参数
            command = ['sudo', '/home/dmdba/dmdbms/bin/DmServiceDMTEST', 'restart']
            subprocess.run(command)
            end_time = time.time()
            # 计算并打印运行时间
            execution_time = end_time - start_time
            print(f"Restart Execution time: {execution_time} seconds")
        except Exception as e:
            print(f'restart Exception {e}')
        # finally:
            # os.chdir(original_directory)  # 恢复到原来的工作目录
        # sudo_exec('sudo /home/dmdba/dmdbms/bin/DmServiceDMTEST restart', 'Czy139123450988')  # 重启 MySQL 服务器
    time.sleep(5)
    return 1


# 将配置写入 MySQL 配置文件
def write_cnf_file(configs):
    """
    Args:
        configs: str, Formatted MySQL Parameters, e.g. "--binlog_size=xxx"  格式化的 MySQL 参数，例如 "--binlog_size=xxx"
    """
    cnf_file = '/dmdata/data/DAMENG/dm.ini'  # MySQL 配置文件路径
    sudo_exec('sudo chmod 777 %s' % cnf_file, 'Czy139123450988')  # 修改配置文件权限为 777
    time.sleep(2)
    # TODO 覆盖相同参数
    with open(cnf_file, 'a') as configfile:
        for param in configs:  # 遍历参数列表
            pair_ = param.split(':')  # 分割参数
            configfile.write(f"{pair_[0]} = {pair_[1]}\n")
        # config_parser.write(open(cnf_file, 'a'))  # 将配置写入配置文件
    sudo_exec('sudo chmod 744 %s' % cnf_file, 'Czy139123450988')  # 修改配置文件权限为 744
    time.sleep(2)


# 启动 XML-RPC 服务器
# def serve():
#
#     server = SimpleXMLRPCServer(('0.0.0.0', 20000))  # 创建 XML-RPC 服务器对象，监听 20000 端口
#     server.register_function(start_mysql)  # 注册启动 MySQL 服务器的函数
#     server.register_function(get_state)  # 注册获取 MySQL 服务器状态的函数
#     server.serve_forever()  # 运行服务器，等待客户端连接
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()  # 创建解析器对象
#     parser.add_argument('--docker', action='store_true')  # 添加参数 --docker
#     opt = parser.parse_args()  # 解析命令行参数
#     if opt.docker:  # 如果解析到opt.docker
#         docker = True
#
#     serve()

