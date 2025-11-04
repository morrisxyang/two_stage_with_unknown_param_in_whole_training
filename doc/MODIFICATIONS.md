# 训练脚本改造说明

## 修改概述
将 `train.py` 改造为使用 `data/custom_datasets_700/knapsack` 目录下的 `.npy` 文件，并支持随机选择700条训练数据。

## 主要修改

### 1. 数据路径更新
- **原路径**: 从父目录的 `data/train_size=700/txt_version/` 读取txt文件
- **新路径**: 从 `data/custom_datasets_700/knapsack/` 读取npy文件
- **输出路径**: 改为 `results/cap={capacity},K={compensation_fee}/` 目录

### 2. 新增数据加载函数
添加了 `load_knapsack_data()` 函数，功能包括：
- 从 `.npy` 文件加载数据（encodings, instances, solutions）
- 使用 `mmap_mode='r'` 高效处理大文件（train_encodings.npy ~703MB）
- 随机选择指定数量的训练样本（默认700条）
- 支持通过 `seed` 参数控制随机采样，确保可重复性

### 3. 数据格式说明
参考 `Data.md` 和 `data.py`：

#### 输入文件格式
- `train_encodings.npy`: (4500, 10, 4096) - 训练集特征向量
- `train_instances.npy`: (4500, 10, 2) - 训练集实例 **[weight, price]**
- `train_sols_cap100.npy`: (4500, 10) - cap=100时的最优解（二值0/1）
- `test_encodings.npy`: (500, 10, 4096) - 测试集特征向量
- `test_instances.npy`: (500, 10, 2) - 测试集实例 **[weight, price]**
- `test_sols_cap100.npy`: (500, 10) - cap=100时的最优解（二值0/1）

#### 数据处理流程
1. 从4500条训练数据中随机选择700条（每次运行使用不同的seed）
2. 将 `(N, 10, 4096)` reshape 为 `(N*10, 4096)` 作为特征
3. 将 instances 的 **[weight, price]** 转换为 **[price, weight]** 以匹配原代码预期
4. 将 `(N, 10, 2)` reshape 为 `(N*10, 2)` 作为目标值

### 4. 主训练循环修改
- 移除了从txt文件读取数据的代码
- 每次测试运行使用不同的seed（0-9）来随机采样700条训练数据
- 数据格式自动转换为训练代码所需的形状

## 使用的数据集
- **Capacity**: 100
- **训练样本**: 从4500条中随机选择700条
- **测试样本**: 全部500条

## 关键参数
```python
capacity = 100              # 背包容量
train_case_num = 700       # 训练样本数量
purchase_fee = 1           # 购买费用
compensation_fee = 5       # 补偿费用
```

## 文件依赖
- `data/custom_datasets_700/knapsack/train_encodings.npy`
- `data/custom_datasets_700/knapsack/train_instances.npy`
- `data/custom_datasets_700/knapsack/train_sols_cap100.npy`
- `data/custom_datasets_700/knapsack/test_encodings.npy`
- `data/custom_datasets_700/knapsack/test_instances.npy`
- `data/custom_datasets_700/knapsack/test_sols_cap100.npy`

## 注意事项
1. 每次训练运行（testTime循环）会使用不同的随机样本，确保模型稳健性
2. 使用内存映射（mmap_mode）加载大文件，避免内存溢出
3. 数据格式转换：instances 从 [weight, price] 转为 [price, weight]
4. 训练结果保存在 `results/cap=100,K=5/` 目录下

