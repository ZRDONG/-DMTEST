# -*- coding: utf-8 -*-
"""
description: MySQL Environment
整合压测工具的代码
"""

import re
import os
import time
import math
# import datetime
# import json
# import threading
# import MySQLdb
import numpy as np
import configs
import environment.utils as utils
import knobs
# import requests
# import psutil
# import server.server as server
import subprocess

TEMP_FILES = "/dmdata/benchmarksql-5.0/run/res_100w_100c_10m_10.log"
BEST_NOW = "/dmdata/benchmarksql-5.0/run/tuner/best_now/"
PROJECT_DIR = "/dmdata/benchmarksql-5.0/run/tuner/"


# TEMP_FILES = "/home/rmw/train_result/tmp/"
# PROJECT_DIR = "/home/rmw/"


class Status(object):
    OK = 'OK'
    FAIL = 'FAIL'
    RETRY = 'RETRY'


class DAMENGEnv(object):

    # method改benchmarksql
    def __init__(self, wk_type='read', method='benchmark', num_other_knobs=0, num_metric=16, alpha=1.0, beta1=0.5,
                 beta2=0.5, time_decay1=1.0, time_decay2=1.0):

        self.db_info = None
        self.wk_type = wk_type
        self.score = 0.0
        self.steps = 0
        self.terminate = False
        self.last_external_metrics = None
        self.default_externam_metrics = None

        self.method = method
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.time_decay_1 = time_decay1
        self.time_decay_2 = time_decay2
        self.num_other_knobs = num_other_knobs
        self.num_metric = num_metric

    @staticmethod
    def _get_external_metrics(path, method='benchmark'):

        # 一个基于TPCC的一个压测评价方法 和benchmarksql是否会有相同 甚至   是否就是benchmarksql
        # 进一步 benchmarksql是否也有一个最终的文档结果
        def parse_benchmark(file_path):
            with open(file_path) as f:
                lines = f.read()
            print(lines)
            temporal_pattern = re.compile(
                r"Measured tpmC \(NewOrders\) = ([\d.]+)\n.*?Measured tpmTOTAL = ([\d.]+)\n.*\n.*\n.*? Transaction Count = (\d+)")
            match = temporal_pattern.search(lines)
            print(match)
            if match:
                # 提取 tpmC、tpmTOTAL 和 Transaction Count 的值
                tpmC = float(match.group(1))
                tpmTOTAL = float(match.group(2))
                transaction_count = int(match.group(3))
                return [tpmC, tpmTOTAL, transaction_count]
            else:
                return None
            #
            # # 提取最后十次测试结果，将tps、latency求和记录下来，除以样本数，得到一个平均的tps、latency
            # for i in temporal[-10:]:
            #     tps += float(i[0])
            #     latency += float(i[2])
            # num_samples = len(temporal[-10:])
            # tps /= num_samples
            # latency /= num_samples
            # # interval
            # tps /= 1
            # return [tps, latency, tps]

        # sysbench的一个评价标准，类似上一个  仅仅正则表达式识别不同，即测试输出结果不同
        # def parse_sysbench(file_path):
        #     with open(file_path) as f:
        #         lines = f.read()
        #     temporal_pattern = re.compile(
        #         "tps: (\d+.\d+) qps: (\d+.\d+) \(r/w/o: (\d+.\d+)/(\d+.\d+)/(\d+.\d+)\)"
        #         " lat \(ms,95%\): (\d+.\d+) err/s: (\d+.\d+) reconn/s: (\d+.\d+)")
        #     temporal = temporal_pattern.findall(lines)
        #     tps = 0
        #     latency = 0
        #     qps = 0
        #
        #     for i in temporal[-10:]:
        #         tps += float(i[0])
        #         latency += float(i[5])
        #         qps += float(i[1])
        #     num_samples = len(temporal[-10:])
        #     tps /= num_samples
        #     qps /= num_samples
        #     latency /= num_samples
        #     return [tps, latency, qps]

        if method == 'benchmark':
            result = parse_benchmark(path)
        # elif method == 'tpcc':
        #     result = parse_tpcc(path)
        # else:
        #     result = parse_sysbench(path)
        return result

#   TODO 多次记录取平均值
    def _get_internal_metrics(self, internal_metrics):
        """
        Args:
            internal_metrics: list,存储内部指标数据的列表
        Return:

        """
        try:
            internal_metrics = utils.get_metrics(self.db_info)
        except Exception as err:
            print("[GET Metrics]Exception:", err)

        return internal_metrics  # internal_metrics : {'name' : value,'name' : value,'name' : value,}

    # 参数metrics即为internal_metrics
    def _post_handle(self, metrics):
        result = np.zeros(self.num_metric)
        print(metrics)

        # def do(metric_name, metric_values):
        #     metric_type = utils.get_metric_type(metric_name)
        #     if metric_type == 'counter':
        #         # 如果是计数型参数则返回数值
        #         return float(metric_values[-1] - metric_values[0])
        #     else:
        #         # 如果是值类型  则返回平均值
        #         return float(sum(metric_values)) / len(metric_values)
        #
        # # 获取所有指标的名称，并按字母顺序排序
        # keys = list(metrics[0].keys())
        # keys.sort()
        # 获取所有指标的名称，并按字母顺序排序
        values = list(metrics.values())

        # 遍历所有指标
        for idx in range(len(values)):
            value = values[idx]  # key = 参数name
            result[idx] = value  # result为value的集合
        return result

    def initialize(self):
        """Initialize the mysql instance environment
        """
        pass

    # 评估knobs的效果  通过tps、latency
    def eval(self, knob):
        """ Evaluate the knobs
        Args:
            knob: dict, mysql parameters
        Returns:
            result: {tps, latency}
        """
        flag = self._apply_knobs(knob)
        if not flag:
            return {"tps": 0, "latency": 0}

        external_metrics, _ = self._get_state(knob, method=self.method)
        return {"tps": external_metrics[0],
                "latency": external_metrics[1]}

    # 读取当前最佳的参数 [tps, latency, tps]
    def _get_best_now(self, filename):
        with open(BEST_NOW + filename) as f:
            lines = f.readlines()
        best_now = lines[0].split(',')
        return [float(best_now[0]), float(best_now[1]), float(best_now[0])]

    # 保存最佳参数   如果新的参数比之前记录的最佳参数更好，则返回 True；否则返回 False
    def record_best(self, external_metrics):
        # 设置保存最佳参数的文件名
        filename = 'bestnow.log'
        best_flag = False
        if os.path.exists(BEST_NOW + filename):
            # 获取当前最佳参数的 TPS 和延迟值
            tpmc_best = external_metrics[0]
            tpmtotal_best = external_metrics[1]
            Transaction = external_metrics[2]
            # rate = 0

            if int(tpmc_best) != 0:
                # 读取文件中记录的最佳参数
                with open(BEST_NOW + filename) as f:
                    lines = f.readlines()
                best_now = lines[0].split(',')
                tpmc_best_now = float(best_now[0])

                # 如果当前的参数比之前记录的最佳参数更好，则更新最佳参数并标记为找到更好的参数
                if tpmc_best > tpmc_best_now:
                    best_flag = True
                    with open(BEST_NOW + filename, 'w') as f:
                        # file.write() 会覆盖原有的内容
                        f.write(str(tpmc_best) + ',' + str(tpmtotal_best) + ',' + str(Transaction))

        else:
            # 如果文件不存在，则创建文件并记录当前的参数作为最佳参数
            file = open(BEST_NOW + filename, 'w+')
            tpmc_best = external_metrics[0]
            tpmtotal_best = external_metrics[1]
            Transaction = external_metrics[2]

            # 计算当前最佳参数的 TPS 和延迟比率
            # 将当前最佳参数写入文件
            file.write(str(tpmc_best) + ',' + str(tpmtotal_best) + ',' + str(Transaction))
        return best_flag

    def step(self, knob):
        """step
        """
        filename = 'bestnow.log'
        # 记录优化步骤的开始时间
        restart_time = utils.time_start()
        # 应用旋钮值，返回是否成功的标志
        flag = self._apply_knobs(knob)
        # 计算优化步骤的时间
        restart_time = utils.time_end(restart_time)

        # 如果应用旋钮值失败，则返回一个负的奖励和其他默认值
        if not flag:
            return -10000000.0, np.array([0] * self.num_metric), True, self.score - 10000000, [0, 0, 0], restart_time

        # 获取当前状态信息
        s = self._get_state(knob, method=self.method)

        # 如果获取状态信息失败，则返回一个负的奖励和其他默认值
        if s is None:
            return -10000000.0, np.array([0] * self.num_metric), True, self.score - 10000000, [0, 0, 0], restart_time

        # 获取外部度量参数和内部度量参数
        external_metrics, internal_metrics = s

        # 根据外部度量计算奖励
        reward = self._get_reward(external_metrics)

        # 记录当前最佳参数，并打印提示信息
        flag = self.record_best(external_metrics)
        if flag == True:
            print('Better performance changed!')
        else:
            print('Performance remained!')
        # get the best performance so far to calculate the reward 获取当前最佳性能以计算奖励
        best_now_performance = self._get_best_now(filename)
        self.last_external_metrics = best_now_performance

        next_state = internal_metrics
        terminate = self._terminate()
        knobs.save_knobs(
            knob=knob,
            metrics=external_metrics,
            knob_file='%ssave_knobs/knob_metric.txt' % PROJECT_DIR
        )
        return reward, next_state, terminate, self.score, external_metrics, restart_time
        # 返回 奖励、下一步状态、终止标志、当前分数、外部度量和重启时间

    def setting(self, knob):
        self._apply_knobs(knob)

    def _get_state(self, knob, method='benchmark'):
        """Collect the Internal State and External State
        """
        # 设置存储的临时文件夹
        filename = TEMP_FILES
        # if not os.path.exists(filename):
        #     os.mkdir(filename)
        # timestamp = int(time.time())  # 这个时间戳表示当前时间与1970年1月1日之间的秒数
        # filename += '%s.txt' % timestamp  # 避免文件名重复
        internal_metrics = {}
        # 获取internal_metrics 即数据库内部参数
        internal_metrics = self._get_internal_metrics(internal_metrics)
        if method == 'benchmark':
            # 目标目录
            run_directory = '/dmdata/benchmarksql-5.0/run'
            # 切换到目标目录并执行命令
            original_directory = os.getcwd()  # 保存当前工作目录
            os.chdir(run_directory)  # 切换到目标目录
            try:
                start_time = time.time()
                # 要执行的命令和参数
                command = ['nohup', './runBenchmark.sh', 'props.dm']
                # 指定输出文件路径
                output_file = 'res_100w_100c_10m_10.log'
                # 使用 subprocess.run() 函数执行命令

                # TODO
                with open(output_file, 'wb') as fp:
                    subprocess.run(command, stdout=fp)


                end_time = time.time()
                # 计算并打印运行时间
                execution_time = end_time - start_time
                print(f"Execution time: {execution_time} seconds")
            finally:
                os.chdir(original_directory)  # 恢复到原来的工作目录
        # 若使用tpcc方法进行压测
        # elif method == 'tpcc':
        #     # 定义一个函数来终止运行的 tpcc 进程
        #     def kill_tpcc():
        #         # 定义一个函数来过滤进程列表中的 tpcc 进程
        #         def _filter_pid(x):
        #             try:
        #                 x = psutil.Process(x)  # 将进程的PID（Process ID）转换为psutil.Process的实例对象，通过实例化这个类可以获取进程的详细信息，比如CPU占用、内存占用等。
        #                 if x.name() == 'tpcc_start':
        #                     return True
        #                 return False
        #             except:
        #                 return False
        #
        #         # 获取当前系统上的所有进程的 PID
        #         pids = psutil.pids()
        #
        #         tpcc_pid = filter(_filter_pid, pids)  # filter(_filter_pid, pids)会返回一个只包含了所有进程中符合条件的PID的迭代器
        #         print tpcc_pid
        #         # 终止所有tpcc进程
        #         for tpcc_pid_i in tpcc_pid:
        #             os.system('kill %s' % tpcc_pid_i)
        #
        #     # 170秒后自动终止tpcc进程
        #     timer = threading.Timer(170, kill_tpcc)
        #     timer.start()
        #
        #     # 运行脚本
        #     os.system('bash %sAutoTuner/scripts/run_tpcc.sh %s %d %s %s' % (PROJECT_DIR,
        #                                                                     self.db_info['host'],
        #                                                                     self.db_info['port'],
        #                                                                     self.db_info['passwd'],
        #                                                                     filename))
        #     time.sleep(10)

        external_metrics = self._get_external_metrics(filename, method)
        internal_metrics = self._post_handle(internal_metrics)

        return external_metrics, internal_metrics  # 返回值external_metrics = [tps, latency, qbs] ,internal_metrics[value1, value2, value3, ...]

    def _apply_knobs(self, knob):
        """ Apply Knobs to the instance
        """
        pass

    # 奖励函数的设置  这个是训练好坏的关键点！
    # delta0(float): 初始状态到当前状态的变化量
    # deltat(float): 上一步到当前状态的变化量
    @staticmethod
    def _calculate_reward(delta0, deltat):
        if delta0 > 0:
            _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
        else:
            _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

        # 如果奖励值大于0且当前状态的变化量小于0，则将奖励值设为0
        if _reward > 0 and deltat < 0:
            _reward = 0
        return _reward

    # 用于获取奖励值
    def _get_reward(self, external_metrics):
        """
        Args:
            external_metrics: list, external metric info, including `tps` and `qps`
        Return:
            reward: float, a scalar reward
        """
        # 打印外部度量参数，用于调试
        print('*****************************')
        print(external_metrics)
        print(self.default_externam_metrics)
        print(self.last_external_metrics)
        print('*****************************')
        # tps
        delta_0_tps = float((external_metrics[0] - self.default_externam_metrics[0])) / self.default_externam_metrics[0]
        delta_t_tps = float((external_metrics[0] - self.last_external_metrics[0])) / self.last_external_metrics[0]

        tps_reward = self._calculate_reward(delta_0_tps, delta_t_tps)

        # # latency
        # delta_0_lat = float((-external_metrics[1] + self.default_externam_metrics[1])) / self.default_externam_metrics[
        #     1]
        # delta_t_lat = float((-external_metrics[1] + self.last_external_metrics[1])) / self.last_external_metrics[1]
        #
        # lat_reward = self._calculate_reward(delta_0_lat, delta_t_lat)

        # 组合 TPS 和延迟的奖励值为总的奖励值，以一定的权重相加
        # reward = tps_reward * 0.4 + 0.6 * lat_reward
        reward = tps_reward
        # 更新总分数
        self.score += reward
        print('$$$$$$$$$$$$$$$$$$$$$$')
        print(tps_reward)
        # print(lat_reward)
        print(reward)
        print('$$$$$$$$$$$$$$$$$$$$$$')

        # 如果奖励值大于0，则将奖励值乘以一个系数
        if reward > 0:
            reward = reward * 1000000
        return reward

    def _terminate(self):
        return self.terminate


class Server(DAMENGEnv):
    """ Build an environment directly on Server
    """

    def __init__(self, wk_type, instance_name):
        # 调用父类构造函数初始化环境
        DAMENGEnv.__init__(self, wk_type)

        # 初始化Server特有的属性
        self.wk_type = wk_type
        self.score = 0.0
        self.steps = 0
        self.terminate = False
        self.last_external_metrics = None
        self.instance_name = instance_name
        self.db_info = configs.instance_config[instance_name]
        self.alpha = 1.0

        # 初始化旋钮参数
        knobs.init_knobs(instance_name, num_more_knobs=0)
        self.default_knobs = knobs.get_init_knobs()

    def initialize(self):
        """ Initialize the environment when an episode starts
        Returns:
            state: np.array, current state
        """

        # 重置分数、步数、终止标志等
        self.score = 0.0
        self.last_external_metrics = []
        self.steps = 0
        self.terminate = False

        # 应用默认的数据库参数
        flag = self._apply_knobs(self.default_knobs)
        i = 0
        # 如果初始化不成功次数过多  则打印日志
        while not flag:
            flag = self._apply_knobs(self.default_knobs)
            i += 1
            if i >= 5:
                print("Initialize: {} times ....".format(i))

        external_metrics, internal_metrics = self._get_state(knob=self.default_knobs, method=self.method)
        self.last_external_metrics = external_metrics
        self.default_externam_metrics = external_metrics
        state = internal_metrics
        knobs.save_knobs(
            self.default_knobs,
            metrics=external_metrics,
            knob_file='%ssave_knobs/knob_metric.txt' % PROJECT_DIR
        )
        return state, external_metrics

    def _apply_knobs(self, knob):
        """ Apply the knobs to the mysql
        Args:
            knob: dict, mysql parameters
        Returns:
            flag: whether the setup is valid
        """
        self.steps += 1
        utils.restart_database(
            instance_name=self.instance_name,
            configuration=knob
        )

        steps = 0
        max_steps = 3
        # 测试数据库连接
        flag = utils.test_dameng(self.instance_name)
        while not flag and steps < max_steps:
            # 获取数据库状态  调试用的？
            _st = utils.get_state()
            time.sleep(5)
            flag = utils.test_dameng(self.instance_name)
            steps += 1

        # 如果最终数据库仍不能正常连接运行，则重新初始化数据库knobs为初始knobs
        if not flag:
            utils.restart_database(
                instance_name=self.instance_name,
                configuration=self.default_knobs
            )
            params = ''

            # 将失败参数记录到日志文件中
            for key in knob.keys():
                params += ' --%s=%s' % (key, knob[key])
            with open('failed.log', 'a+') as f:
                f.write('{}\n'.format(params))
            return False
        else:
            return True


DockerServer = Server
