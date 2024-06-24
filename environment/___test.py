import os
import time
import platform
import subprocess
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
    with open(output_file, 'wb') as fp:
        subprocess.run(command, stdout=fp)

    end_time = time.time()

    # 计算并打印运行时间
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
finally:
    os.chdir(original_directory)  # 恢复到原来的工作目录