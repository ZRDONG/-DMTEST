# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import time
import pickle
import logging
import datetime


def time_start():
    return time.time()


def time_end(start):
    end = time.time()
    delay = end - start
    return delay


def get_timestamp():
    """
    获取UNIX时间戳
    """
    return int(time.time())


def time_to_str(timestamp):
    """
    将时间戳转换成[YYYY-MM-DD HH:mm:ss]格式
    """
    return datetime.datetime.\
        fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class Logger:

    def __init__(self, name, log_file=''):
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        self.logger.addHandler(sh)
        if len(log_file) > 0:
            self.log2file = True
        else:
            self.log2file = False

    def _write_file(self, msg):
        if self.log2file:
            with open(self.log_file, 'a+') as f:
                f.write(msg + '\n')

    def get_timestr(self):
        timestamp = get_timestamp()
        date_str = time_to_str(timestamp)
        return date_str

    def warn(self, msg):
        msg = "%s[WARN] %s" % (self.get_timestr(), msg)
        self.logger.warning(msg)
        self._write_file(msg)

    def info(self, msg):
        msg = "%s[INFO] %s" % (self.get_timestr(), msg)
        #self.logger.info(msg)
        self._write_file(msg)

    def error(self, msg):
        msg = "%s[ERROR] %s" % (self.get_timestr(), msg)
        self.logger.error(msg)
        self._write_file(msg)


def save_state_actions(state_action, filename):

    f = open(filename, 'wb')
    pickle.dump(state_action, f)
    f.close()


def plot_tpmC(tpmC, epoch):

    if len(tpmC) > 10 or epoch > 10:
        tpmC_plot = tpmC[-10:]
        epoch_plot = list(range(epoch - 9, epoch + 1))

    else:
        tpmC_plot = tpmC
        epoch_plot = list(range(0, len(tpmC)))

    # if len(tpmC) == 0:
    #     plt.xlim(0, 1)
    # else:
    #     plt.xlim(epoch_plot[0], epoch_plot[-1] + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_plot, tpmC_plot, marker='o', linestyle='-', color='b', label='')
    plt.title('TpmC (Last 10 Entries)')
    plt.ylim(0, 100000)
    plt.xlim(0,10)
    plt.xlabel('Epoch')
    plt.ylabel('TpmC')
    # plt.legend()

    # 显示网格
    plt.grid(True)  # 启用网格线

    plt.savefig('plot/tpmC.png')
    # 显示图形
    plt.show()  # 显示图表

def plot_knob(knob):
    knob_last_ten = knob[-10:]
    result = {}
    for d in knob_last_ten:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    print(result)
    for key, values in result.items():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(values) + 1), values, marker='o', linestyle='-', label=key)
        plt.title(f'{key}')
        # plt.xlabel('')
        plt.ylabel(f'{key}')
        # plt.legend()
        plt.grid(True)
        plt.xticks(range(0, 10, 1))
        plt.yticks(range(min(values), max(values) + 1, int((min(values)+max(values))/10) + 1))
        plt.savefig(f'plot/{key}.png')  # 保存图表到指定路径
        plt.show()  # 显示图表