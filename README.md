# Task-aware + Sample-aware + Choquet-inspired Multi-Agent Voting Demo

完整可运行的 Python / PyTorch demo，用来验证一个通用多智能体分类决策框架：

- 固定 agent 专家池
- task-aware relevance
- sample-aware relevance
- 可训练的 single-agent 权重
- 可训练的 pairwise agent interaction 权重
- Choquet-inspired 非加性聚合

demo 不调用真实大模型 API。5 个 agent 用轻量规则模拟不同专家视角，重点是验证“多 agent 输出 + 任务相关性 + 样本相关性 + 组合交互权重 + 反向训练”的机制。

## 项目结构

```text
choquet_agent_vote_demo/
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── data/
│   └── toy_data.csv
├── src/
│   ├── dataset.py
│   ├── agents.py
│   ├── embeddings.py
│   ├── choquet_layer.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
```

## 方法思想

框架面对不同任务时，使用同一组固定 agent：

1. Semantic Agent：关注语义、上下文、隐含含义
2. Emotion Agent：关注情绪、态度、讽刺、主观性
3. Intention Agent：关注诱导、误导、操纵、吸引点击
4. Lexical Agent：关注关键词、夸张词、标点和表层模式
5. Consistency Agent：关注标题、正文、图片描述之间的一致性或矛盾

每个 agent 输入 `text` 和 `task_description`，输出二分类概率、置信度和简短 explanation。

当前 toy 数据支持两个任务：

- `clickbait detection`
- `implicit sentiment detection`

首次运行时，如果 `data/toy_data.csv` 不足 80 行，程序会自动生成约 160 条 toy samples。

## 为什么不是普通投票

普通 majority voting 或 average probability voting 假设所有 agent 在所有任务、所有样本上同等重要。

这个 demo 显式建模：

- 不同任务下 agent 重要性不同，例如 clickbait 更依赖 Intention / Lexical，implicit sentiment 更依赖 Emotion / Semantic。
- 不同样本会触发不同 agent，例如带有 `however / actually / 然而 / 其实` 的文本会提高 Consistency Agent 的相关性。
- agent 之间不是简单相加，有些组合会产生 synergy，有些组合可能冗余。

## 为什么是 Choquet-inspired

完整 Choquet integral 适合表达非加性聚合：一个 agent 的贡献不只取决于自己，也取决于它和其他 agent 的组合关系。

在多智能体投票里，这对应：

- inter-agent interaction
- synergy / redundancy modeling
- pairwise capacity-like weights
- 非简单平均的决策融合

本项目中的 `ChoquetInspiredVotingLayer` 学习两类贡献：

```text
single contribution = sum_i w_i * p_i
pair contribution   = sum_{i<j} w_ij * interaction(p_i, p_j)
```

默认 `interaction(p_i, p_j) = p_i * p_j`，代码中也标注了可替换为 `min(p_i, p_j)` 的位置。

## 为什么采用 2-additive approximation

完整 Choquet capacity 需要 `2^K` 个 capacity。K 变大后复杂度很高。

本 demo 使用 2-additive approximation，只建模：

- single agent capacity
- pairwise agent capacity

这样复杂度从指数级降到 `O(K^2)`，更适合轻量多智能体投票实验。它不是严格完整 Choquet integral 的数学实现，但保留了 Choquet-inspired 的核心：非加性聚合、组合交互、synergy / redundancy、capacity-like pair weights。

## 可训练权重

single weight：

```text
single_weight_i = softmax(
    a_i * task_relevance_i
  + b_i * sample_relevance_i
  + c_i * confidence_i
  + bias_i
)
```

pair weight：

```text
pair_weight_ij = sigmoid(
    u_ij * task_relevance_i * task_relevance_j
  + v_ij * sample_relevance_i * sample_relevance_j
  + r_ij * agreement_ij
  + m_ij
)
```

这些参数都在 PyTorch 中通过 `CrossEntropyLoss` 反向传播训练。默认 agent 是规则型固定专家，训练主要更新 Choquet-inspired voting layer。

## 如何运行

```bash
pip install -r requirements.txt
python main.py
```

默认 CPU 可运行。随机种子固定在 `config.py` 中。

## 输出结果说明

`python main.py` 会输出：

1. 数据集规模和 train / valid / test 划分
2. 每 5 个 epoch 的训练 loss、验证 accuracy、macro F1
3. 最佳模型保存路径
4. 四种方法对比：
   - Majority Voting
   - Average Probability Voting
   - Dynamic Single-Agent Weighting
   - Choquet-inspired Pairwise Voting
5. 不同任务下的平均 single agent weights
6. 不同任务下 top pairwise weights
7. 若干样本的决策轨迹：
   - text
   - gold label
   - each agent prediction / confidence / explanation
   - single weights
   - top pairwise weights
   - final prediction

同时会保存两个输出文件：

- `outputs/best_choquet_model.pt`：PyTorch 二进制 checkpoint，用于程序加载，直接用文本编辑器打开会像乱码，这是正常现象。
- `outputs/model_summary.json`：UTF-8 可读摘要，适合在 PyCharm 中查看，包括 agent 名称、pair 名称、训练参数、任务级平均权重。

## 后续如何替换真实 LLM agent

可以保持 `ChoquetInspiredVotingLayer` 不变，只替换 `src/agents.py` 中的 agent 实现：

- 把规则 agent 替换为 GPT / Gemini / Qwen / 本地 LLM API
- 每个 agent 使用不同 role prompt
- 每个 agent 返回标准化概率、置信度、解释文本
- 继续使用 task relevance、sample relevance 和 Choquet-inspired voting layer 聚合

也可以让 agent 输出更多类别，只需调整 `NUM_CLASSES` 和数据标签即可。
