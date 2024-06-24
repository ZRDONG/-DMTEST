# -*- coding: utf-8 -*-
"""

Prioritized Replay Memory  优先级经验回放记忆(Prioritized Replay Memory)
"""
import random
import pickle
import numpy as np


class SumTree(object):
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)  # 定义存储优先级的数组
        self.data = np.zeros(capacity, dtype=object)  # 定义存储数据的数组
        self.num_entries = 0

    # 递归函数，用于更新父节点的优先级
    def _propagate(self, idx, change):
        parent = int((idx - 1) / 2)  # 计算父节点的索引
        self.tree[parent] += change  # 更新父节点的优先级
        if parent != 0:
            self._propagate(parent, change)

    # 递归函数，用于根据优先级检索叶子节点的索引
    def _retrieve(self, idx, s):
        left = 2 * idx + 1  # 计算左子节点的索引
        right = left + 1  # 计算右子节点的索引

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:  # 若优先级小于左子节点，则向左子树查找
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])  # 否则向右子树查找

    # 返回优先级总和（根节点的值）
    def total(self):
        return self.tree[0]

    # 向 SumTree 中添加新的优先级和数据
    def add(self, p, data):
        idx = self.write + self.capacity - 1  # 计算叶子节点的索引

        self.data[self.write] = data  # 将数据存储在对应索引处
        self.update(idx, p)  # 更新优先级树中的优先级

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.num_entries < self.capacity:
            self.num_entries += 1

    # 更新 SumTree 中指定节点的优先级
    def update(self, idx, p):
        change = p - self.tree[idx]  # 计算优先级变化量

        self.tree[idx] = p  # 更新节点的优先级
        self._propagate(idx, change)  # 更新父节点的优先级

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return [idx, self.tree[idx], self.data[data_idx]]


# 定义优先级经验回放记忆类
class PrioritizedReplayMemory(object):

    def __init__(self, capacity):
        self.tree = SumTree(capacity)  # 初始化优先级树
        self.capacity = capacity  # 记忆库容量
        self.e = 0.01  # 优先级的偏移量，用于确保每个样本都有一定的优先级
        self.a = 0.6  # 控制优先级的程度，通常取值在[0, 1]之间
        self.beta = 0.4  # 用于调整优先级采样偏差的参数
        self.beta_increment_per_sampling = 0.001  # 每次采样时增加的 beta 值

    # 根据误差计算优先级
    def _get_priority(self, error):
        return (error + self.e) ** self.a

    # 向记忆库中添加样本和对应的优先级
    def add(self, error, sample):
        # (s, a, r, s, t)
        p = self._get_priority(error)   # 计算优先级
        self.tree.add(p, sample)  # 向优先级树中添加优先级和样本

    # 返回记忆库中存储的样本数量
    def __len__(self):
        return self.tree.num_entries

    # 从记忆库中采样一批样本
    def sample(self, n):
        batch = []
        idxs = []  # 存储采样的样本在优先级树中的索引
        segment = self.tree.total() / n  # 计算每个样本的优先级区间大小
        priorities = []  # 存储采样的样本对应的优先级

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i  # 计算优先级区间的起始值
            b = segment * (i + 1)  # 计算优先级区间的结束值

            s = random.uniform(a, b)  # 在优先级区间内随机采样
            (idx, p, data) = self.tree.get(s)  # 从优先级树中检索样本和对应的优先级
            priorities.append(p)  # 存储优先级
            batch.append(data)  # 存储样本
            idxs.append(idx)  # 存储样本在优先级树中的索引
        return batch, idxs  # 返回采样的样本和对应的索引

        # sampling_probabilities = priorities / self.tree.total()
        # is_weight = np.power(self.tree.num_entries * sampling_probabilities, -self.beta)
        # is_weight /= is_weight.max()

    # 更新指定样本的优先级
    def update(self, idx, error):
        p = self._get_priority(error)  # 计算优先级
        self.tree.update(idx, p)  # 更新优先级树中的优先级

    # 将优先级经验回放记忆保存到文件中
    def save(self, path):
        f = open(path, 'wb')
        pickle.dump({"tree": self.tree}, f)
        f.close()

    # 从文件中加载优先级经验回放记忆
    def load_memory(self, path):
        with open(path, 'rb') as f:
            _memory = pickle.load(f)
        self.tree = _memory['tree']

