# -*- coding: utf-8 -*-
"""
Train the model
"""

import os
import sys
import utils
import pickle
import argparse
sys.path.append('../')
import models
import numpy as np
import environment


def generate_knob(action, method):
    if method == 'ddpg':
        return environment.gen_continuous(action)
    else:
        raise NotImplementedError('Not Implemented')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tencent', action='store_true', help='Use Tencent Server')
    parser.add_argument('--params', type=str, default='', help='Load existing parameters')
    parser.add_argument('--workload', type=str, default='read', help='Workload type [`read`, `write`, `readwrite`]')
    parser.add_argument('--instance', type=str, default='dameng', help='Choose MySQL Instance')
    parser.add_argument('--method', type=str, default='ddpg', help='Choose Algorithm to solve [`ddpg`,`dqn`]')
    parser.add_argument('--memory', type=str, default='', help='add replay memory')
    parser.add_argument('--noisy', action='store_true', help='use noisy linear layer')
    parser.add_argument('--other_knob', type=int, default=0, help='Number of other knobs')
    parser.add_argument('--batch_size', type=int, default=2, help='Training Batch Size')
    parser.add_argument('--epoches', type=int, default=5000000, help='Training Epoches')
    parser.add_argument('--benchmark', type=str, default='sysbench', help='[sysbench, tpcc]')
    parser.add_argument('--metric_num', type=int, default=16, help='metric nums')
    parser.add_argument('--default_knobs', type=int, default=197, help='default knobs')
    opt = parser.parse_args()

    # Create Environment
    # if opt.tencent:
    #     env = environment.TencentServer(
    #         wk_type=opt.workload,
    #         instance_name=opt.instance,
    #         method=opt.benchmark,
    #         num_metric=opt.metric_num,
    #         num_other_knobs=opt.other_knob)
    # else:
    env = environment.Server(wk_type=opt.workload, instance_name=opt.instance)

    # Build models
    ddpg_opt = dict()
    ddpg_opt['tau'] = 0.00001
    ddpg_opt['alr'] = 0.00001
    ddpg_opt['clr'] = 0.00001
    ddpg_opt['model'] = opt.params
    n_states = opt.metric_num
    gamma = 0.9
    memory_size = 100000
    num_actions = opt.default_knobs + opt.other_knob
    ddpg_opt['gamma'] = gamma
    ddpg_opt['batch_size'] = opt.batch_size
    ddpg_opt['memory_size'] = memory_size

    model = models.DDPG(
        n_states=n_states,
        n_actions=num_actions,
        opt=ddpg_opt,
        mean_var_path='mean_var.pkl',
        ouprocess=not opt.noisy
    )

    if not os.path.exists('log'):
        os.mkdir('log')

    if not os.path.exists('save_memory'):
        os.mkdir('save_memory')

    if not os.path.exists('save_knobs'):
        os.mkdir('save_knobs')

    if not os.path.exists('save_state_actions'):
        os.mkdir('save_state_actions')

    if not os.path.exists('model_params'):
        os.mkdir('model_params')

    # 使用当前方法和时间戳生成表达式名称
    expr_name = 'train_{}_{}'.format(opt.method, str(utils.get_timestamp()))

    # 初始化日志记录器，记录训练过程中的日志信息
    logger = utils.Logger(
        name=opt.method,   # 使用方法名称作为日志记录器的名称
        log_file='log/{}.log'.format(expr_name)  # 日志文件路径，基于生成的表达式名称
    )

    # 如果存在其他旋钮，则发出警告记录
    if opt.other_knob != 0:
        logger.warn('USE Other Knobs')

    current_knob = environment.get_init_knobs()

    # OUProcess
    origin_sigma = 0.20
    sigma = origin_sigma

    # decay rate
    sigma_decay_rate = 0.9
    step_counter = 0
    train_step = 0

    # 累积损失列表，用于记录评论家和演员的损失
    accumulate_loss = [0, 0]

    # 用于记录状态动作对中奖励较高的状态动作
    fine_state_actions = []

    # 如果存在记忆，则加载记忆
    if len(opt.memory) > 0:
        model.replay_memory.load_memory(opt.memory)
        print("Load Memory: {}".format(len(model.replay_memory)))

    # time for every step  每一步的时间
    step_times = []
    # time for training  训练时间
    train_step_times = []
    # time for setup, restart, test  设置、重启、测试时间
    env_step_times = []
    # restart time  重启时间
    env_restart_times = []
    # choose_action_time  选择动作时间
    action_step_times = []

    for episode in range(opt.epoches):
        # 初始化环境，获取当前状态和初始指标
        current_state, initial_metrics = env.initialize()
        logger.info("\n[Env initialized][Metric tpmC: {} tpmTotal: {} Transaction Count: {}]".format(
            initial_metrics[0], initial_metrics[1], initial_metrics[2]))

        # 重置模型噪声
        model.reset(sigma)
        t = 0  # 初始化步数计数器

        # 循环执行动作直到环境结束
        while True:
            step_time = utils.time_start()  # 记录每步开始的时间
            state = current_state  # 获取当前状态
            if opt.noisy:
                model.sample_noise()  # 若使用有噪声的线性层，对模型进行噪声采样
            action_step_time = utils.time_start()  # 记录选择动作开始的时间
            action = model.choose_action(state)  # 根据当前状态选择动作
            action_step_time = utils.time_end(action_step_time)  # 记录选择动作结束的时间

            current_knob = generate_knob(action, 'ddpg')  # 生成当前动作对应的旋钮配置
            logger.info("[ddpg] Action: {}".format(action))  # 记录选择的动作

            env_step_time = utils.time_start()  # 记录环境执行动作开始的时间
            reward, state_, done, score, metrics, restart_time = env.step(current_knob)  # 执行动作并获取环境反馈
            env_step_time = utils.time_end(env_step_time)
            logger.info(
                "\n[{}][Episode: {}][Step: {}][Metric tpmC:{} tpmTotal:{} Transaction Count:{}]Reward: {} Score: {} Done: {}".format(
                    opt.method, episode, t, metrics[0], metrics[1], metrics[2], reward, score, done
                ))
            env_restart_times.append(restart_time)  # 记录重启时间

            next_state = state_  # 获取下一个状态

            model.add_sample(state, action, reward, next_state, done)  # 将状态动作奖励样本添加到模型中

            if reward > 10:
                fine_state_actions.append((state, action))

            current_state = next_state  # 更新当前状态
            train_step_time = 0.0  # 初始化训练步骤时间
            # 如果记忆中样本数大于批量大小，则进行模型更新
            if len(model.replay_memory) > opt.batch_size:
                losses = []
                train_step_time = utils.time_start()
                for i in range(2):
                    losses.append(model.update())
                    train_step += 1
                train_step_time = utils.time_end(train_step_time)/2.0  # 计算更新模型的平均时间

                # 累积评论家和演员网络的损失
                accumulate_loss[0] += sum([x[0] for x in losses])
                accumulate_loss[1] += sum([x[1] for x in losses])
                logger.info('[{}][Episode: {}][Step: {}] Critic: {} Actor: {}'.format(
                    opt.method, episode, t, accumulate_loss[0] / train_step, accumulate_loss[1] / train_step
                ))

            # all_step time
            step_time = utils.time_end(step_time)
            step_times.append(step_time)
            # env_step_time
            env_step_times.append(env_step_time)
            # training step time
            train_step_times.append(train_step_time)
            # action step times
            action_step_times.append(action_step_time)

            logger.info("[{}][Episode: {}][Step: {}] step: {}s env step: {}s train step: {}s restart time: {}s "
                        "action time: {}s"
                        .format(opt.method, episode, t, step_time, env_step_time, train_step_time,restart_time,
                                action_step_time))

            logger.info("[{}][Episode: {}][Step: {}][Average] step: {}s env step: {}s train step: {}s "
                        "restart time: {}s action time: {}s"
                        .format(opt.method, episode, t, np.mean(step_time), np.mean(env_step_time),
                                np.mean(train_step_time), np.mean(restart_time), np.mean(action_step_times)))

            t = t + 1
            step_counter += 1

            # save replay memory
            if step_counter % 10 == 0:
                model.replay_memory.save('save_memory/{}.pkl'.format(expr_name))
                utils.save_state_actions(fine_state_actions, 'save_state_actions/{}.pkl'.format(expr_name))
                # sigma = origin_sigma*(sigma_decay_rate ** (step_counter/10))

            # save network
            if step_counter % 5 == 0:
                model.save_model('model_params', title='{}_{}'.format(expr_name, step_counter))

            if done or score < -50:
                break







