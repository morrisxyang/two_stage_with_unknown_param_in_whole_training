### datasets/knapsack 中的文件说明
- `raw_sentences.txt`
  - 文本版的物品描述，共约 50k 行（例如 “The Clouded Flail of Hannibal (29 pounds, $299)”）。
  - 与下方两个 50k 行/样本的数组按行对齐。

- `sentences.npy`
  - 形状：(50000, 4096)，dtype=float32。
  - 对应 `raw_sentences.txt` 中每个物品描述的语义向量嵌入（4096 维）。

- `weights_prices.npy`
  - 形状：(50000, 2)，dtype=float64。
  - 每行一个物品的 `[weight, price]`，与 `sentences.npy` 的行一一对应。

- `train_instances.npy`
  - 形状：(4500, 10, 2)，dtype=float64。
  - 训练集的实例集合；每个实例有 10 个物品，每个物品为 `[weight, price]`。

- `test_instances.npy`
  - 形状：(500, 10, 2)，dtype=float64。
  - 测试集的实例集合；同上，每实例 10 个 `[weight, price]`。

- `train_encodings.npy`
  - 形状：(4500, 10, 4096)，dtype=float32。
  - 与 `train_instances.npy` 对齐的 10 个物品的文本嵌入（每个物品 4096 维）。

- `test_encodings.npy`
  - 形状：(500, 10, 4096)，dtype=float32。
  - 与 `test_instances.npy` 对齐的文本嵌入。

- `train_sols.npy`
  - 形状：(4500, 10)，dtype=float64。
  - 训练集每个实例的最优背包选择解，二值 0/1（选择与否）。在加载器中会被归一化到 [-0.5, 0.5] 区间。

- `test_sols.npy`
  - 形状：(500, 10)，dtype=float64。
  - 测试集的最优二值解；同样会在加载时归一化。

### 训练时实际用到的文件
- `data.py` 的 `knapsack_dataloader` 读取：
  - 特征：`train_encodings.npy`, `test_encodings.npy`
  - 标签：`train_sols.npy`, `test_sols.npy`（随后用 `[lb=0, ub=1]` 归一化）
- 其余文件用于数据来源与可解释性（如原始文本、全量物品库的权重与价格、显式的 `[weight, price]` 实例），
 便于检查和分析，但不直接被该加载器调用。