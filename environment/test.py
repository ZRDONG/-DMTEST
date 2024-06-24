import re
import subprocess
import time
import os
import utils as util_environment
from tuner import utils as utils

KNOBS = {
    'ADAPTIVE_NPLN_FLAG': ['enum', [0, 1, 2, 3]],
    # 'BDTA_SIZE': ['integer', [1, 10000, 300]],
    # 'BTR_SPLIT_MODE': ['enum', [0, 1]],
    # 'BUFFER': ['integer', [8, 1048576, 1000]],
    # 'BUFFER_POOLS': ['integer', [1, 512, 19]],
    # 'CACHE_POOL_SIZE': ['integer', [1, 67108864, 100]],
    # 'CASE_WHEN_CVT_IFUN': ['integer', [0, 15, 9]],
    # 'CKPT_DIRTY_PAGES': ['integer', [0, 4294967294, 0]],
    # 'CKPT_FLUSH_PAGES': ['integer', [1000, 100000, 1000]],
    # 'CKPT_FLUSH_RATE': ['integer', [0, 100, 5]],
    # 'CKPT_INTERVAL': ['integer', [0, 2147483647, 180]],
    # 'CKPT_RLOG_SIZE': ['integer', [0, 4294967294, 128]],
    # 'CKPT_WAIT_PAGES': ['integer', [1, 65534, 1024]],
    # 'COMM_VALIDATE': ['enum', [0, 2, 1]],
    # 'COMPLEX_VIEW_MERGING': ['enum', [1, 2, 0]],
    # 'DECIMAL_FIX_STORAGE': ['enum', [1, 0]],
    # 'DICT_BUF_SIZE': ['integer', [1, 2048, 50]],
    # 'DIRECT_IO': ['enum', [1, 2, 0]],
    # 'ENABLE_FREQROOTS': ['enum', [1, 2, 3, 0]],
    # 'ENABLE_HASH_JOIN': ['enum', [0, 1]],
    # 'ENABLE_IN_VALUE_LIST_OPT': ['integer', [0, 15, 6]],
    # 'ENABLE_INDEX_JOIN': ['enum', [0, 1]],
    # 'ENABLE_MERGE_JOIN': ['enum', [0, 1]],
    # 'ENABLE_MONITOR': ['enum', [0, 1]],
    # 'ENHANCED_BEXP_TRANS_GEN': ['enum', [0, 1, 2, 3]],
    # 'FAST_POOL_PAGES': ['integer', [0, 99999999, 3000]],
    # 'FAST_RELEASE_SLOCK': ['enum', [0, 1]],
    # 'FAST_ROLL_PAGES': ['integer', [0, 9999999, 1000]],
    # 'FAST_RW_LOCK': ['enum', [0, 2, 1]],
    # 'FIRST_ROWS': ['integer', [1, 1000000, 100]],
    # 'FORCE_FLUSH_PAGES': ['integer', [0, 1000, 8]],
    # 'GROUP_OPT_FLAG': ['integer', [0, 63, 4]],
    # 'HAGR_BLK_SIZE': ['integer', [1, 50, 2]],
    # 'HAGR_BUF_GLOBAL_SIZE': ['integer', [10, 1000000, 5000]],
    # 'HAGR_BUF_SIZE': ['integer', [2, 500000, 500]],
    # 'HASH_PLL_OPT_FLAG': ['integer', [0, 63, 0]],
    # 'HIO_THR_GROUPS': ['integer', [1, 512, 2]],
    # 'HJ_BLK_SIZE': ['integer', [1, 50, 2]],
    # 'HJ_BUF_GLOBAL_SIZE': ['integer', [10, 500000, 5000]],
    # 'HJ_BUF_SIZE': ['integer', [2, 100000, 1000]],
    # 'HUGE_BUFFER': ['integer', [80, 1048576, 80]],
    # 'HUGE_BUFFER_POOLS': ['integer', [1, 512, 4]],
    # 'HUGE_MEMORY_PERCENTAGE': ['integer', [1, 100, 50]],
    # 'INDEX_SKIP_SCAN_RATE': ['enum', [1, 0]],
    # 'IO_THR_GROUPS': ['integer', [1, 512, 8]],
    # 'JOIN_HASH_SIZE': ['integer', [1, 250000000, 500000]],
    # 'LIKE_OPT_FLAG': ['integer', [0, 127, 127]],
    # 'LOCK_DICT_OPT': ['enum', [0, 1, 2]],
    # 'MEMORY_EXTENT_SIZE': ['integer', [1, 10240, 1]],
    # 'MEMORY_POOL': ['integer', [64, 67108864, 500]],
    # 'MEMORY_TARGET': ['integer', [0, 67108864, 15000]],
    # 'MSG_COMPRESS_TYPE': ['enum', [0, 1, 2]],
    # 'NBEXP_OPT_FLAG': ['integer', [0, 7, 7]],
    # 'NONREFED_SUBQUERY_AS_CONST': ['enum', [1, 0]],
    # 'NOWAIT_WHEN_UNIQUE_CONFLICT': ['enum', [1, 0]],
    # 'OPTIMIZER_AGGR_GROUPBY_ELIM': ['enum', [0, 1]],
    # 'OUTER_CVT_INNER_PULL_UP_COND_FLAG': ['enum', [1, 2, 3]],
    # 'PARALLEL_PURGE_FLAG': ['enum', [1, 0]],
    # 'PHC_MODE_ENFORCE': ['integer', [0, 15, 0]],
    # 'PSEG_RECV': ['enum', [0, 1, 2, 3]],
    # 'PURGE_DEL_OPT': ['enum', [1, 2, 0]],
    # 'PURGE_WAIT_TIME': ['integer', [0, 60000, 500]],
    # 'RECYCLE': ['integer', [8, 1048576, 300]],
    # 'RECYCLE_POOLS': ['integer', [1, 512, 19]],
    # 'REFED_SUBQ_CROSS_FLAG': ['enum', [0, 1]],
    # 'RLOG_BUF_SIZE': ['integer', [1, 20480, 1024]],
    # 'RLOG_PARALLEL_ENABLE': ['enum', [1, 2, 0]],
    # 'RLOG_POOL_SIZE': ['integer', [1, 4096, 256]],
    # 'SESS_POOL_SIZE': ['integer', [16, 1048576, 64]],
    # 'SESS_POOL_TARGET': ['integer', [0, 10485760, 16384]],
    # 'SORT_BUF_SIZE': ['integer', [1, 2048, 20]],
    # 'SPEED_SEMI_JOIN_PLAN': ['integer', [0, 31, 9]],
    # 'TRX_VIEW_MODE': ['enum', [0, 1]],
    # 'TRX_VIEW_SIZE': ['integer', [16, 65000, 512]],
    # 'UNDO_EXTENT_NUM': ['integer', [1, 8192, 4]],
    # 'UNDO_RETENTION': ['integer', [0, 86400, 90]],
    # 'UPD_DEL_OPT': ['enum', [0, 1, 2]],
    # 'VIEW_FILTER_MERGING': ['integer', [0, 511, 138]],
    # 'VM_MEM_HEAP': ['enum', [1, 2, 0]],
    # 'VM_POOL_SIZE': ['integer', [32, 1048576, 64]],
    # 'VM_POOL_TARGET': ['integer', [0, 10485760, 16384]],
    # 'WORK_THRD_RESERVE_SIZE': ['integer', [200, 1024, 200]],
    # 'WORK_THRD_STACK_SIZE': ['integer', [1024, 32768, 8192]],
    # 'WORKER_THREADS': ['integer', [1, 64, 16]],
}



TEMP_FILES = "/dmdata/benchmarksql-5.0/run/res_100w_100c_10m_10.log"


def auto(instance_name, knob):
    util_environment.restart_database(
        instance_name=instance_name,
        configuration=knob
    )

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


def get_external_metrics(path, method='benchmark'):
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

    if method == 'benchmark':
        result = parse_benchmark(path)
    # elif method == 'tpcc':
    #     result = parse_tpcc(path)
    # else:
    #     result = parse_sysbench(path)
    return result


if __name__ == '__main__':
    if not os.path.exists('log'):
        os.mkdir('log')

    # 使用当前方法和时间戳生成表达式名称
    expr_name = 'train_{}_{}'.format('ddpg', str(utils.get_timestamp()))

    # 初始化日志记录器，记录训练过程中的日志信息
    logger = utils.Logger(
        name='ddpg',  # 使用方法名称作为日志记录器的名称
        log_file='log/{}.log'.format(expr_name)  # 日志文件路径，基于生成的表达式名称
    )

    for name, value in KNOBS.items():
        knobs = {}
        knob_value = value[1]
        knobs[name] = knob_value[-1]

        knobs[name] = knobs[name] * 1
        print("{}:{}".format(name ,knobs[name]))
        auto('dameng', knobs)


        filename = TEMP_FILES
        external_metrics = get_external_metrics(filename, 'benchmark')
        logger.info("[{}][Knob: {}] tpmC: {}"
                    .format('ddpg', name, external_metrics[0]))
