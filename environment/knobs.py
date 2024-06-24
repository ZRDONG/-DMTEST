# -*- coding: utf-8 -*-
"""
desciption: Knob information

"""

import configs

# 设置内存大小：700GB
memory_size = 360 * 1024 * 1024  # 这行代码计算内存大小，单位为千字节（KB）。360*1024*1024 的值等于 700GB。

# 设置磁盘大小
disk_size = 8 * 1024 * 1024 * 1024
instance_name = ''

KNOBS = [
    # 'ADAPTIVE_NPLN_FLAG',
    # 'BATCH_PARAM_OPT',
    # 'BDTA_SIZE',
    # 'BTR_SPLIT_MODE',
    # 'BUFFER',
    # 'BUFFER_POOLS',
    # 'CACHE_POOL_SIZE',
    # 'CASE_WHEN_CVT_IFUN',
    # 'CKPT_DIRTY_PAGES',
    # 'CKPT_FLUSH_PAGES',
    # 'CKPT_FLUSH_RATE',
    # 'CKPT_INTERVAL',
    # 'CKPT_RLOG_SIZE',
    # 'CKPT_WAIT_PAGES',
    # 'CLT_CONST_TO_PARAM',
    # 'CNNTB_HASH_TABLE_SIZE',
    # 'CNNTB_OPT_FLAG',
    # 'COMM_VALIDATE',
    # 'COMPLEX_VIEW_MERGING',
    # 'CTE_OPT_FLAG',
    # 'DECIMAL_FIX_STORAGE',
    # 'DEL_HP_OPT_FLAG',
    # 'DICT_BUF_SIZE',
    # 'DIRECT_IO',:
    # 'DIST_HASH_ALGORITHM_FLAG',
    # 'DISTINCT_USE_INDEX_SKIP',
    # 'DTABLE_PULLUP_FLAG',
    # 'DYNAMIC_CALC_NODES',
    # 'ENABLE_ADJUST_DIST_COST',
    # 'ENABLE_ADJUST_NLI_COST',
    # 'ENABLE_CHOOSE_BY_ESTIMATE_FLAG',
    # 'ENABLE_DIST_IN_SUBQUERY_OPT',
    # 'ENABLE_FREQROOTS',
    # 'ENABLE_HASH_JOIN',
    # 'ENABLE_IN_VALUE_LIST_OPT',
    # 'ENABLE_INDEX_FILTER',
    # 'ENABLE_INDEX_JOIN',
    # 'ENABLE_JOIN_FACTORIZATION',
    # 'ENABLE_MERGE_JOIN',
    # 'ENABLE_MONITOR',
    # 'ENABLE_NEST_LOOP_JOIN_CACHE',
    # 'ENABLE_PARTITION_WISE_OPT',
    # 'ENABLE_RQ_TO_INV',
    # 'ENABLE_RQ_TO_NONREF_SPL',
    # 'ENABLE_RQ_TO_SPL',
    # 'ENHANCED_BEXP_TRANS_GEN',
    # 'ENHANCED_SUBQ_MERGING',
    # 'FAST_POOL_PAGES',
    # 'FAST_RELEASE_SLOCK',
    # 'FAST_ROLL_PAGES',
    # 'FAST_RW_LOCK',
    # 'FILTER_PUSH_DOWN',
    # 'FINS_LOCK_MODE',
    # 'FIRST_ROWS',
    # 'FORCE_FLUSH_PAGES',
    # 'FROM_OPT_FLAG',
    # 'FTAB_MEM_SIZE',
    # 'GLOBAL_RTREE_BUF_SIZE',
    # 'GROUP_OPT_FLAG',
    # 'HAGR_BLK_SIZE',
    # 'HAGR_BUF_GLOBAL_SIZE',
    # 'HAGR_BUF_SIZE',
    # 'HAGR_DISTINCT_HASH_TABLE_SIZE',
    # 'HAGR_HASH_ALGORITHM_FLAG',
    # 'HAGR_HASH_SIZE',
    # 'HASH_CMP_OPT_FLAG',
    # 'HASH_PLL_OPT_FLAG',
    # 'HFINS_MAX_THRD_NUM',
    # 'HFINS_PARALLEL_FLAG',
    # 'HFS_CACHE_SIZE',
    # 'HIO_THR_GROUPS',
    # 'HJ_BLK_SIZE',
    # 'HJ_BUF_GLOBAL_SIZE',
    # 'HJ_BUF_SIZE',
    # 'HLSM_FLAG',
    # 'HUGE_BUFFER',
    # 'HUGE_BUFFER_POOLS',
    # 'HUGE_MEMORY_PERCENTAGE',
    # 'IN_LIST_AS_JOIN_KEY',
    # 'INDEX_SKIP_SCAN_RATE',
    # 'IO_THR_GROUPS',
    # 'JOIN_HASH_SIZE',
    # 'KEEP',
    # 'LDIS_NEW_FOLD_FUN',
    # 'LIKE_OPT_FLAG',
    # 'LOCK_DICT_OPT',
    # 'LOCK_TID_MODE',
    # 'LOCK_TID_UPGRADE',
    # 'MAX_N_GRP_PUSH_DOWN',
    # 'MAX_OPT_N_TABLES',
    # 'MAX_PARALLEL_DEGREE',
    # 'MAX_PHC_BE_NUM',
    # 'MEMORY_EXTENT_SIZE',
    # 'MEMORY_N_POOLS',
    # 'MEMORY_POOL',
    # 'MEMORY_TARGET',
    # 'MMT_FLAG',
    # 'MMT_GLOBAL_SIZE',
    # 'MMT_SIZE',
    # 'MSG_COMPRESS_TYPE',
    # 'MTAB_MEM_SIZE',
    # 'MULTI_IN_CVT_EXISTS',
    # 'MULTI_PAGE_GET_NUM',
    # 'NBEXP_OPT_FLAG',
    # 'NEW_MOTION',
    # 'NONCONST_OR_CVT_IN_LST_FLAG',
    # 'NONREFED_SUBQUERY_AS_CONST',
    # 'NOWAIT_WHEN_UNIQUE_CONFLICT',
    # 'OLAP_FLAG',
    # 'OP_SUBQ_CVT_IN_FLAG',
    # 'OPT_OR_FOR_HUGE_TABLE_FLAG',
    # 'OPTIMIZER_AGGR_GROUPBY_ELIM',
    # 'OPTIMIZER_DYNAMIC_SAMPLING',
    # 'OPTIMIZER_MAX_PERM',
    # 'OPTIMIZER_MODE',
    # 'OPTIMIZER_OR_NBEXP',
    # 'OR_CVT_HTAB_FLAG',
    # 'OR_NBEXP_CVT_CASE_WHEN_FLAG',
    # 'OUTER_CVT_INNER_PULL_UP_COND_FLAG',
    # 'OUTER_JOIN_FLATING_FLAG',
    # 'OUTER_JOIN_INDEX_OPT_FLAG',
    # 'OUTER_OPT_NLO_FLAG',
    # 'PARALLEL_MODE_COMMON_DEGREE',
    # 'PARALLEL_POLICY',
    # 'PARALLEL_PURGE_FLAG',
    # 'PARALLEL_THRD_NUM',
    # 'PARTIAL_JOIN_EVALUATION_FLAG',
    # 'PHC_MODE_ENFORCE',
    # 'PLACE_GROUP_BY_FLAG',
    # 'PLN_DICT_HASH_THRESHOLD',
    # 'PRJT_REPLACE_NPAR',
    # 'PSEG_RECV',
    # 'PURGE_DEL_OPT',
    # 'PURGE_WAIT_TIME',
    # 'PUSH_SUBQ',
    # 'RECYCLE',
    # 'RECYCLE_POOLS',
    # 'REFED_EXISTS_OPT_FLAG',
    # 'REFED_OPS_SUBQUERY_OPT_FLAG',
    # 'REFED_SUBQ_CROSS_FLAG',
    # 'RLOG_BUF_SIZE',
    # 'RLOG_PARALLEL_ENABLE',
    # 'RLOG_POOL_SIZE',
    # 'RS_BDTA_BUF_SIZE',
    # 'RS_BDTA_FLAG',
    # 'RT_HEAP_TARGET',
    # 'SEL_ITEM_HTAB_FLAG',
    # 'SEL_RATE_EQU',
    # 'SEL_RATE_SINGLE',
    # 'SESS_POOL_SIZE',
    # 'SESS_POOL_TARGET',
    # 'SINGLE_RTREE_BUF_SIZE',
    # 'SORT_ADAPTIVE_FLAG',
    # 'SORT_BLK_SIZE',
    # 'SORT_BUF_GLOBAL_SIZE',
    # 'SORT_BUF_SIZE',
    # 'SORT_FLAG',
    # 'SORT_OPT_SIZE',
    # 'SPEED_SEMI_JOIN_PLAN',
    # 'SPIN_TIME',
    # 'STAT_ALL',
    # 'STAT_COLLECT_SIZE',
    # 'STAT_OPT_FLAG',
    # 'SUBQ_CVT_SPL_FLAG',
    # 'SUBQ_EXP_CVT_FLAG',
    # 'TASK_THREADS',
    # 'TEMP_SIZE',
    # 'TMP_DEL_OPT',
    # 'TOP_DIS_HASH_FLAG',
    # 'TOP_ORDER_ESTIMATE_CARD',
    # 'TOP_ORDER_OPT_FLAG',
    # 'TRX_CMTARR_SIZE',
    # 'TRX_DICT_LOCK_NUM',
    # 'TRX_VIEW_MODE',
    # 'TRX_VIEW_SIZE',
    # 'TSORT_OPT',
    # 'UNDO_EXTENT_NUM',
    # 'UNDO_RETENTION',
    # 'UPD_DEL_OPT',
    # 'USE_FJ_REMOVE_TABLE_FLAG',
    # 'USE_FK_REMOVE_TABLES_FLAG',
    # 'USE_HAGR_FLAG',
    # 'USE_HTAB',
    # 'USE_INDEX_SKIP_SCAN',
    # 'USE_REFER_TAB_ONLY',
    # 'VIEW_FILTER_MERGING',
    # 'VIEW_OPT_FLAG',
    # 'VIEW_PULLUP_FLAG',
    # 'VIEW_PULLUP_MAX_TAB',
    # 'VM_MEM_HEAP',
    # 'VM_POOL_SIZE',
    # 'VM_POOL_TARGET',
    # 'VM_STACK_SIZE',
    # 'WORK_THRD_RESERVE_SIZE',
    # 'WORK_THRD_STACK_SIZE',
    # 'WORKER_CPU_PERCENT',
    # 'WORKER_THREADS',
]

KNOB_DETAILS = None
EXTENDED_KNOBS = None
num_knobs = len(KNOBS)


# instance导入连接，
def init_knobs(instance, num_more_knobs):
    global instance_name
    global memory_size
    global disk_size
    global KNOB_DETAILS
    global EXTENDED_KNOBS
    instance_name = instance
    # TODO: Test the request
    use_request = False  # 标志变量，用于控制是否进行请求测试
    if use_request:
        # 如果实例名称中包含'tencent'，则调用一个用于获取腾讯云实例信息的函数
        # if instance_name.find('tencent') != -1:
        #     memory_size, disk_size = utils.get_tencent_instance_info(instance_name)
        # else:
        # 否则，从配置中获取实例的内存大小
        memory_size = configs.instance_config[instance_name]['memory']
        # disk_size = configs.instance_config[instance_name]['disk']
    else:
        memory_size = 1024
        # 如果 use_request 为 False，则执行以下代码块
        # 磁盘大小直接使用预先设置的配置数据
        # memory_size = configs.instance_config[instance_name]['memory']
        # disk_size = configs.instance_config[instance_name]['disk']

    # 设置旋钮相关可调节范围、数值参数的细节信息  参数形式为[类型[min，max，默认值]]
    KNOB_DETAILS = {
        # 'ADAPTIVE_NPLN_FLAG': ['enum', [0, 1, 2, 3]],
        # 'BATCH_PARAM_OPT': ['enum', [0, 1]],
        # 'BDTA_SIZE': ['integer', [1, 10000, 300]],
        # 'BTR_SPLIT_MODE': ['enum', [0, 1]],
        # 'BUFFER': ['integer', [13000, 13000, 13000]],
        # 'BUFFER_POOLS': ['integer', [1, 512, 19]],
        # 'CACHE_POOL_SIZE': ['integer', [1, 67108864, 100]],
        # 'CASE_WHEN_CVT_IFUN': ['integer', [0, 15, 9]],
        # 'CKPT_DIRTY_PAGES': ['integer', [0, 4294967294, 0]],
        # 'CKPT_FLUSH_PAGES': ['integer', [1000, 100000, 1000]],
        # 'CKPT_FLUSH_RATE': ['integer', [0, 100, 5]],
        # 'CKPT_INTERVAL': ['integer', [0, 2147483647, 180]],
        # 'CKPT_RLOG_SIZE': ['integer', [0, 4294967294, 128]],
        # 'CKPT_WAIT_PAGES': ['integer', [1, 65534, 1024]],
        # 'CLT_CONST_TO_PARAM': ['enum', [0, 1]],
        # 'CNNTB_HASH_TABLE_SIZE': ['integer', [100, 100000000, 100]],
        # 'CNNTB_OPT_FLAG': ['integer', [0, 255, 0]],
        # 'COMM_VALIDATE': ['enum', [0, 2, 1]],
        # 'COMPLEX_VIEW_MERGING': ['enum', [1, 2, 0]],
        # 'CTE_OPT_FLAG': ['enum', [0, 1]],
        # 'DECIMAL_FIX_STORAGE': ['enum', [1, 0]],
        # 'DEL_HP_OPT_FLAG': ['integer', [0, 31, 0]],
        # 'DICT_BUF_SIZE': ['integer', [1, 2048, 50]],
        # 'DIRECT_IO': ['enum', [1, 2, 0]],
        # 'DIST_HASH_ALGORITHM_FLAG': ['enum', [1, 0]],
        # 'DISTINCT_USE_INDEX_SKIP': ['enum', [0, 1, 2]],
        # 'DTABLE_PULLUP_FLAG': ['enum', [0, 1]],
        # 'DYNAMIC_CALC_NODES': ['enum', [1, 0]],
        # 'ENABLE_ADJUST_DIST_COST': ['enum', [1, 2, 0]],
        # 'ENABLE_ADJUST_NLI_COST': ['enum', [0, 1]],
        # 'ENABLE_CHOOSE_BY_ESTIMATE_FLAG': ['integer', [0, 7, 0]],
        # 'ENABLE_DIST_IN_SUBQUERY_OPT': ['integer', [0, 7, 0]],
        # 'ENABLE_FREQROOTS': ['enum', [1, 2, 3, 0]],
        # 'ENABLE_HASH_JOIN': ['enum', [0, 1]],
        # 'ENABLE_IN_VALUE_LIST_OPT': ['integer', [0, 15, 6]],
        # 'ENABLE_INDEX_FILTER': ['enum', [1, 2, 0]],
        # 'ENABLE_INDEX_JOIN': ['enum', [0, 1]],
        # 'ENABLE_JOIN_FACTORIZATION': ['enum', [1, 0]],
        # 'ENABLE_MERGE_JOIN': ['enum', [0, 1]],
        # 'ENABLE_MONITOR': ['enum', [0, 1]],
        # 'ENABLE_NEST_LOOP_JOIN_CACHE': ['enum', [1, 2, 3, 0]],
        # 'ENABLE_PARTITION_WISE_OPT': ['enum', [1, 0]],
        # 'ENABLE_RQ_TO_INV': ['enum', [1, 2, 0]],
        # 'ENABLE_RQ_TO_NONREF_SPL': ['integer', [0, 7, 0]],
        # 'ENABLE_RQ_TO_SPL': ['enum', [0, 1]],
        # 'ENHANCED_BEXP_TRANS_GEN': ['enum', [0, 1, 2, 3]],
        # 'ENHANCED_SUBQ_MERGING': ['enum', [1, 2, 3, 0]],
        # 'FAST_POOL_PAGES': ['integer', [0, 99999999, 3000]],
        # 'FAST_RELEASE_SLOCK': ['enum', [0, 1]],
        # 'FAST_ROLL_PAGES': ['integer', [0, 9999999, 1000]],
        # 'FAST_RW_LOCK': ['enum', [0, 2, 1]],
        # 'FILTER_PUSH_DOWN': ['integer', [0, 15, 0]],
        # 'FINS_LOCK_MODE': ['enum', [0, 2, 1]],
        # 'FIRST_ROWS': ['integer', [1, 1000000, 100]],
        # 'FORCE_FLUSH_PAGES': ['integer', [0, 1000, 8]],
        # 'FROM_OPT_FLAG': ['enum', [1, 0]],
        # 'FTAB_MEM_SIZE': ['integer', [0, 64 * 1024, 0]],
        # 'GLOBAL_RTREE_BUF_SIZE': ['integer', [0, 100000, 100]],
        # 'GROUP_OPT_FLAG': ['integer', [0, 63, 4]],
        # 'HAGR_BLK_SIZE': ['integer', [1, 50, 2]],
        # 'HAGR_BUF_GLOBAL_SIZE': ['integer', [10, 1000000, 5000]],
        # 'HAGR_BUF_SIZE': ['integer', [2, 500000, 500]],
        # 'HAGR_DISTINCT_HASH_TABLE_SIZE': ['integer', [10000, 100000000, 10000]],
        # 'HAGR_HASH_ALGORITHM_FLAG': ['enum', [1, 0]],
        # 'HAGR_HASH_SIZE': ['integer', [10000, 100000000, 100000]],
        # 'HASH_CMP_OPT_FLAG': ['integer', [0, 15, 0]],
        # 'HASH_PLL_OPT_FLAG': ['integer', [0, 63, 0]],
        # 'HFINS_MAX_THRD_NUM': ['integer', [4, 200, 100]],
        # 'HFINS_PARALLEL_FLAG': ['enum', [1, 2, 0]],
        # 'HFS_CACHE_SIZE': ['integer', [160, 2000, 160]],
        # 'HIO_THR_GROUPS': ['integer', [1, 512, 2]],
        # 'HJ_BLK_SIZE': ['integer', [1, 50, 2]],
        # 'HJ_BUF_GLOBAL_SIZE': ['integer', [10, 500000, 5000]],
        # 'HJ_BUF_SIZE': ['integer', [2, 100000, 1000]],
        # 'HLSM_FLAG': ['enum', [2, 3, 1]],
        # 'HUGE_BUFFER': ['integer', [80, 10000, 80]],
        # 'HUGE_BUFFER_POOLS': ['integer', [1, 512, 4]],
        # 'HUGE_MEMORY_PERCENTAGE': ['integer', [1, 100, 50]],
        # 'IN_LIST_AS_JOIN_KEY': ['enum', [1, 0]],
        # 'INDEX_SKIP_SCAN_RATE': ['enum', [1, 0]],
        # 'IO_THR_GROUPS': ['integer', [1, 512, 8]],
        # 'JOIN_HASH_SIZE': ['integer', [1, 250000000, 500000]],
        # 'KEEP': ['integer', [8, 1048576, 8]],
        # 'LDIS_NEW_FOLD_FUN': ['enum', [1, 0]],
        # 'LIKE_OPT_FLAG': ['integer', [0, 127, 127]],
        # 'LOCK_DICT_OPT': ['enum', [0, 1, 2]],
        # 'LOCK_TID_MODE': ['enum', [0, 1]],
        # 'LOCK_TID_UPGRADE': ['enum', [1, 0]],
        # 'MAX_N_GRP_PUSH_DOWN': ['integer', [1, 1024, 5]],
        # 'MAX_OPT_N_TABLES': ['integer', [3, 8, 6]],
        # 'MAX_PARALLEL_DEGREE': ['integer', [0, 128, 1]],
        # 'MAX_PHC_BE_NUM': ['integer', [512, 20480000, 512]],
        # 'MEMORY_EXTENT_SIZE': ['integer', [1, 1024, 1]],
        # 'MEMORY_N_POOLS': ['integer', [0, 2048, 1]],
        # 'MEMORY_POOL': ['integer', [64, 10000, 500]],
        # 'MEMORY_TARGET': ['integer', [0, 15000, 15000]],
        # 'MMT_FLAG': ['enum', [2, 1]],
        # 'MMT_GLOBAL_SIZE': ['integer', [10, 1000000]],
        # 'MMT_SIZE': ['integer', [0, 64, 0]],
        # 'MSG_COMPRESS_TYPE': ['enum', [0, 1, 2]],
        # 'MTAB_MEM_SIZE': ['integer', [1, 1048576, 8]],
        # 'MULTI_IN_CVT_EXISTS': ['enum', [1, 0]],
        # 'MULTI_PAGE_GET_NUM': ['integer', [1, 64, 1]],
        # 'NBEXP_OPT_FLAG': ['integer', [0, 7, 7]],
        # 'NEW_MOTION': ['enum', [1, 0]],
        # 'NONCONST_OR_CVT_IN_LST_FLAG': ['enum', [1, 0]],
        # 'NONREFED_SUBQUERY_AS_CONST': ['enum', [1, 0]],
        # 'NOWAIT_WHEN_UNIQUE_CONFLICT': ['enum', [1, 0]],
        # 'OLAP_FLAG': ['enum', [0, 1, 2]],
        # 'OP_SUBQ_CVT_IN_FLAG': ['enum', [0, 1]],
        # 'OPT_OR_FOR_HUGE_TABLE_FLAG': ['enum', [0, 1]],
        # 'OPTIMIZER_AGGR_GROUPBY_ELIM': ['enum', [0, 1]],
        # 'OPTIMIZER_DYNAMIC_SAMPLING': ['integer', [0, 12, 0]],
        # 'OPTIMIZER_MAX_PERM': ['integer', [1, 4294967294, 7200]],
        # 'OPTIMIZER_MODE': ['enum', [0, 1]],
        # 'OPTIMIZER_OR_NBEXP': ['integer', [0, 31, 0]],
        # 'OR_CVT_HTAB_FLAG': ['enum', [0, 2, 1]],
        # 'OR_NBEXP_CVT_CASE_WHEN_FLAG': ['enum', [1, 0]],
        # 'OUTER_CVT_INNER_PULL_UP_COND_FLAG': ['enum', [1, 2, 3]],
        # 'OUTER_JOIN_FLATING_FLAG': ['enum', [1, 0]],
        # 'OUTER_JOIN_INDEX_OPT_FLAG': ['enum', [1, 0]],
        # 'OUTER_OPT_NLO_FLAG': ['integer', [0, 7, 0]],
        # 'PARALLEL_MODE_COMMON_DEGREE': ['integer', [1, 1024, 1]],
        # 'PARALLEL_POLICY': ['enum', [1, 2, 0]],
        # 'PARALLEL_PURGE_FLAG': ['enum', [1, 0]],
        # 'PARALLEL_THRD_NUM': ['integer', [1, 1024, 10]],
        # 'PARTIAL_JOIN_EVALUATION_FLAG': ['enum', [0, 1]],
        # 'PHC_MODE_ENFORCE': ['integer', [0, 15, 0]],
        # 'PLACE_GROUP_BY_FLAG': ['enum', [1, 2, 3, 0]],
        # 'PLN_DICT_HASH_THRESHOLD': ['integer', [1, 67108864, 20]],
        # 'PRJT_REPLACE_NPAR': ['enum', [0, 1]],
        # 'PSEG_RECV': ['enum', [0, 1, 2, 3]],
        # 'PURGE_DEL_OPT': ['enum', [1, 2, 0]],
        # 'PURGE_WAIT_TIME': ['integer', [0, 60000, 500]],
        # 'PUSH_SUBQ': ['integer', [0, 7, 0]],
        # 'RECYCLE': ['integer', [8, 10000, 300]],
        # 'RECYCLE_POOLS': ['integer', [1, 512, 19]],
        # 'REFED_EXISTS_OPT_FLAG': ['enum', [1, 2,  3, 0]],
        # 'REFED_OPS_SUBQUERY_OPT_FLAG': ['integer', [0, 7, 0]],
        # 'REFED_SUBQ_CROSS_FLAG': ['enum', [0, 1]],
        # 'RLOG_BUF_SIZE': ['integer', [1, 20480, 1024]],
        # 'RLOG_PARALLEL_ENABLE': ['enum', [1, 2, 0]],
        # 'RLOG_POOL_SIZE': ['integer', [1, 4096, 256]],
        # 'RS_BDTA_BUF_SIZE': ['integer', [8, 32768, 32]],
        # 'RS_BDTA_FLAG': ['enum', [1, 2, 0]],
        # 'RT_HEAP_TARGET': ['integer', [8192, 10485760, 8192]],
        # 'SEL_ITEM_HTAB_FLAG': ['integer', [0, 15, 0]],
        # 'SEL_RATE_EQU': ['enum', [1, 0]],
        # 'SEL_RATE_SINGLE': ['enum', [1, 0]],
        # 'SESS_POOL_SIZE': ['integer', [16, 1048576, 64]],
        # 'SESS_POOL_TARGET': ['integer', [0, 10485760, 16384]],
        # 'SINGLE_RTREE_BUF_SIZE': ['integer', [0, 1000, 10]],
        # 'SORT_ADAPTIVE_FLAG': ['enum', [1, 2, 3, 0]],
        # 'SORT_BLK_SIZE': ['integer', [1, 50, 1]],
        # 'SORT_BUF_GLOBAL_SIZE': ['integer', [10, 4294967294, 1000]],
        # 'SORT_BUF_SIZE': ['integer', [1, 2048, 20]],
        # 'SORT_FLAG': ['enum', [1, 2, 3, 0]],
        # 'SORT_OPT_SIZE': ['integer', [0, 1024, 0]],
        # 'SPEED_SEMI_JOIN_PLAN': ['integer', [0, 31, 9]],
        # 'SPIN_TIME': ['integer', [0, 4000, 4000]],
        # 'STAT_ALL': ['enum', [1, 2, 3, 4, 0]],
        # 'STAT_COLLECT_SIZE': ['integer', [0, 10000000, 10000]],
        # 'STAT_OPT_FLAG': ['enum', [1, 0]],
        # 'SUBQ_CVT_SPL_FLAG': ['integer', [0, 31, 1]],
        # 'SUBQ_EXP_CVT_FLAG': ['integer', [0, 255, 0]],
        # 'TASK_THREADS': ['integer', [1, 1000, 16]],
        # 'TEMP_SIZE': ['integer', [10, 1048576, 10]],
        # 'TMP_DEL_OPT': ['enum', [0, 1]],
        # 'TOP_DIS_HASH_FLAG': ['enum', [0, 2, 1]],
        # 'TOP_ORDER_ESTIMATE_CARD': ['integer', [0, 4294967294, 300]],
        # 'TOP_ORDER_OPT_FLAG': ['integer', [0, 15, 0]],
        # 'TRX_CMTARR_SIZE': ['integer', [3, 1000, 10]],
        # 'TRX_DICT_LOCK_NUM': ['integer', [1, 100000, 64]],
        # 'TRX_VIEW_MODE': ['enum', [0, 1]],
        # 'TRX_VIEW_SIZE': ['integer', [16, 65000, 512]],
        # 'TSORT_OPT': ['enum', [0, 1]],
        # 'UNDO_EXTENT_NUM': ['integer', [1, 8192, 4]],
        # 'UNDO_RETENTION': ['integer', [0, 86400, 90]],
        # 'UPD_DEL_OPT': ['enum', [0, 1, 2]],
        # 'USE_FJ_REMOVE_TABLE_FLAG': ['enum', [1, 0]],
        # 'USE_FK_REMOVE_TABLES_FLAG': ['enum', [0, 1]],
        # 'USE_HAGR_FLAG': ['enum', [1, 0]],
        # 'USE_HTAB': ['enum', [1, 0]],
        # 'USE_INDEX_SKIP_SCAN': ['enum', [1, 2, 0]],
        # 'USE_REFER_TAB_ONLY': ['enum', [1, 0]],
        # 'VIEW_FILTER_MERGING': ['integer', [0, 511, 138]],
        # 'VIEW_OPT_FLAG': ['enum', [0, 1]],
        # 'VIEW_PULLUP_FLAG': ['integer', [0, 63, 0]],
        # 'VIEW_PULLUP_MAX_TAB': ['integer', [1, 16, 7]],
        # 'VM_MEM_HEAP': ['enum', [1, 2, 0]],
        # 'VM_POOL_SIZE': ['integer', [32, 1048576, 64]],
        # 'VM_POOL_TARGET': ['integer', [0, 10485760, 16384]],
        # 'VM_STACK_SIZE': ['integer', [64, 262144, 256]],
        # 'WORK_THRD_RESERVE_SIZE': ['integer', [200, 1024, 200]],
        # 'WORK_THRD_STACK_SIZE': ['integer', [1024, 32768, 8192]],
        # 'WORKER_CPU_PERCENT': ['integer', [0, 100, 0]],
        # 'WORKER_THREADS': ['integer', [1, 64, 16]],
    }

    print("Instance: %s Memory: %s" % (instance_name, memory_size))


def get_init_knobs():
    knobs = {}

    # 对参数分别进行初始化，设置参数值为默认值
    for name, value in KNOB_DETAILS.items():
        knob_value = value[1]
        knobs[name] = knob_value[-1]

    return knobs


def gen_continuous(action):
    knobs = {}

    # 遍历配置初始的参数列表，KNOBS为初始自带的参数列表，客户未进行过相关添加参数工作
    for idx in range(len(KNOBS)):
        # 获取参数的名称
        name = KNOBS[idx]
        # 获取参数的描述信息,包括类型和取值范围,value应该为一个list,形式为:[类型,[min,max,default]]
        value = KNOB_DETAILS[name]

        knob_type = value[0]  # 类型
        knob_value = value[1]  # list[min,max,default]
        min_value = knob_value[0]

        if knob_type == 'integer':
            max_val = knob_value[1]
            eval_value = int(max_val * action[idx])  # action的形式是一个numpy数组，用于存储旋钮调节的数值大小，且大小被限制在了（0,1）
            eval_value = max(eval_value, min_value)  # 因为通过设置action的大小介于（0,1）因此不会超过最高值，因此只考虑大于最小值
        else:
            enum_size = len(knob_value)
            enum_index = int(enum_size * action[idx])  # 通过设置action*可选参数长度为int值来选取 偏向于动作发生的一个参数  有点会的
            enum_index = min(enum_size - 1, enum_index)  # 因为通过设置action的大小介于（0,1）因此不会出现负数，因此只考虑小于最大值
            eval_value = knob_value[enum_index]

        # if name == 'innodb_log_file_size':
        #    max_val = disk_size / knobs['innodb_log_files_in_group']
        #    eval_value = int(max_val * action[idx])
        #    eval_value = max(eval_value, min_value)

        # if name == 'binlog_cache_size':
        #    if knobs['binlog_cache_size'] > knobs['max_binlog_cache_size']:
        #        max_val = knobs['max_binlog_cache_size']
        #        eval_value = int(max_val * action[idx])
        #        eval_value = max(eval_value, min_value)

        # 设置参数值
        knobs[name] = eval_value

    # if 'tmp_table_size' in knobs.keys():
    # tmp_table_size
    # max_heap_table_size = knobs.get('max_heap_table_size', -1)
    # act_value = knobs['tmp_table_size']/EXTENDED_KNOBS['tmp_table_size'][1][1]
    # max_val = min(EXTENDED_KNOBS['tmp_table_size'][1][1], max_heap_table_size)\
    # if max_heap_table_size > 0 else EXTENDED_KNOBS['tmp_table_size'][1][1]
    # eval_value = int(max_val * act_value)
    # eval_value = max(eval_value, EXTENDED_KNOBS['tmp_table_size'][1][0])
    # knobs['tmp_table_size'] = eval_value

    return knobs


# 简单的存储knob、metrics的一个方法
def save_knobs(knob, metrics, knob_file):
    """ Save Knobs and their metrics to files
    Args:
        knob: dict, knob content
        metrics: list, tps and latency
        knob_file: str, file path
    """
    # format: tps, latency, knobstr: [#knobname=value#]
    knob_strs = []
    for kv in knob.items():
        knob_strs.append('{}:{}'.format(kv[0], kv[1]))
    result_str = '{},{},{},'.format(metrics[0], metrics[1], metrics[2])
    knob_str = "#".join(knob_strs)
    result_str += knob_str

    with open(knob_file, 'a+') as f:
        f.write(result_str + '\n')
