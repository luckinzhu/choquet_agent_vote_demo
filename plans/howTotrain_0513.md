我来帮你分析系统的训练和测试逻辑。让我先查看相关的核心文件。

让我再查看模型和数据集相关的文件，以便全面了解训练和测试逻辑。

现在让我查看 Choquet 层和 agents 的实现，以便完整理解整个系统。

现在我已经全面了解了系统的训练和测试逻辑。让我为你详细解释：

## 系统训练和测试逻辑

### **整体架构**

这是一个基于 **Choquet 投票机制的多智能体混合模型**，核心思想是：
- **Agent（智能体）固定不变**：5个专家agent（Semantic、Emotion、Intention、Lexical、Consistency）的输出是固定的，不参与训练
- **只训练聚合层**：只有 ChoquetInspiredVotingLayer 是可训练的，负责学习如何动态加权融合各个agent的输出

---

### **一、数据流程**

1. **数据加载与生成** (`src/dataset.py`)
   - 使用 `ensure_toy_data()` 加载或生成 toy 数据集
   - 包含两个任务：点击诱导检测 (clickbait) 和隐式情感检测 (implicit_sentiment)
   - 每条数据包含：`task_name`, `task_description`, `text`, `label`

2. **数据划分** (`load_and_split()`)
   - 按 70% 训练集、15% 验证集、15% 测试集划分
   - 使用分层抽样保持类别平衡

---

### **二、Agent 系统**

支持三种后端模式（通过 `AGENT_BACKEND` 环境变量控制）：

1. **Rule 模式**：5个基于规则的固定专家
2. **LLM 模式**：5个基于大模型的专家（需要 API key）
3. **Hybrid 模式**：LLM + Rule fallback

每个 Agent 输出：
- `probs`: [class_0概率, class_1概率]
- `confidence`: 置信度
- `explanation`: 解释文本

---

### **三、特征工程**

在 `model.make_inputs()` 中为每个样本提取：

1. **Agent 输出**：5个agent的概率分布和置信度
2. **TF-IDF 相关性**：
   - `task_relevance`: 任务描述与agent描述的相似度
   - `sample_relevance`: 文本内容与agent描述的相似度

---

### **四、训练流程** (`main.py` → `train_choquet_model()`)

#### **1. 初始化阶段**
```python
# main.py line 216
model = MultiAgentChoquetModel(num_classes=2, device="cpu", agent_backend=backend)
```


#### **2. LLM 预热缓存** (line 246-250)
- 如果使用 LLM/Hybrid 模式，先对所有数据运行一次 agent
- 将结果缓存到 `llm_cache.json`，避免训练时重复调用 API

#### **3. 训练循环** (`src/train.py`)

**优化器设置**：
```python
optimizer = torch.optim.AdamW(
    model.layer.parameters(),  # 只优化 Choquet 层
    lr=0.035,
    weight_decay=1e-4
)
```


**Epoch 循环** (默认35轮)：
```python
for epoch in range(1, epochs + 1):
    # 1. 训练一个epoch
    train_loss = _run_epoch(model, train_df, optimizer, batch_size, device)
    
    # 2. 验证集评估
    valid_metrics = evaluate_model(model, valid_df, batch_size, device, use_pairwise=True)
    
    # 3. 保存最佳模型（基于 macro_f1）
    if valid_metrics["macro_f1"] > best_f1:
        best_f1 = valid_metrics["macro_f1"]
        torch.save(best_state, model_path)
```


**单个 Batch 的训练** (`_run_epoch()`)：
```python
# 1. 获取 agent 输出和相关性特征
inputs = model.make_inputs(batch)

# 2. 前向传播：通过 Choquet 层得到 logits
logits = model.logits_from_inputs(inputs, use_pairwise=True, details=False)

# 3. 计算 CrossEntropyLoss
loss = criterion(logits, labels)

# 4. 反向传播更新 Choquet 层参数
optimizer.zero_grad()
loss.backward()
optimizer.step()
```


---

### **五、Choquet 层的可训练参数** (`src/choquet_layer.py`)

#### **Singleton Weights（单个agent权重）**
```python
single_logits = a * task_relevance + b * sample_relevance + c * confidence + bias
single_weights = softmax(single_logits)
```

- `a`, `b`, `c`, `single_bias`: 可学习参数

#### **Pairwise Weights（agent对交互权重）**
```python
pair_logits = u * (task_i * task_j) + v * (sample_i * sample_j) + r * agreement + bias
pair_weights = sigmoid(pair_logits)
```

- `u`, `v`, `r`, `pair_bias`: 可学习参数
- `agreement`: agent间预测的一致性（1 - L1距离）

#### **最终得分**
```python
scores = single_contribution + pair_scale * pair_contribution
logits = scores * logit_scale + logit_bias
```


---

### **六、测试与评估流程**

#### **1. 测试集评估** (`main.py` line 272)
```python
test_metrics = evaluate_model(model, test_df, BATCH_SIZE, DEVICE, use_pairwise=True)
```

计算指标：loss, accuracy, precision, recall, macro_f1

#### **2. 多方法对比** (`compare_methods()`)
比较4种聚合策略：
- **Majority Voting**: 多数投票
- **Average Probability Voting**: 平均概率投票
- **Dynamic Single-Agent Weighting**: 只使用单agent权重的动态加权
- **Choquet-inspired Pairwise Voting**: 完整的Choquet投票（最终模型）

#### **3. 可解释性分析**
- `print_interpretability_summary()`: 打印各任务和agent的平均权重
- `print_sample_decisions()`: 展示具体样本的决策过程
- `export_readable_model_summary()`: 导出JSON格式的模型摘要

---

### **七、关键设计特点**

1. **冻结Agent，训练聚合器**：
   - Agent输出固定，确保可解释性
   - 只训练轻量级的Choquet层（O(K²)复杂度）

2. **2-additive Choquet近似**：
   - 完整Choquet容量需要 2^K 个集合函数
   - 这里只用 singleton + pairwise，降低到 O(K²)

3. **上下文感知的动态权重**：
   - 权重不是固定的，而是根据 task_relevance、sample_relevance、confidence 动态计算

4. **成对交互建模**：
   - 捕捉agent之间的协同（reinforce）或冗余（overlap）关系

5. **缓存机制**：
   - LLM输出缓存避免重复调用API，加速训练

---

### **八、输出文件**

每次运行会在 `outputs/runs/XXXX_timestamp_backend_model/` 下生成：
- `best_choquet_model.pt`: 最佳模型checkpoint
- `model_summary.json`: 可读的模型权重摘要
- `run_config.json`: 运行配置快照
- `run_result.json`: 运行结果（验证/测试指标）

根目录 `outputs/` 下会同步最新的模型和摘要文件。

---

这就是整个系统的训练和测试逻辑！核心思想是通过学习 **动态的、上下文感知的agent权重和交互关系**，而不是简单地平均或投票。