完成了：默认仍是原来的 `inspired`，新增 `discrete_2additive` 只作为可选对照模式。

**1. CPU/GPU 调查与修复**
当前项目不需要 GPU。依据：

- 训练的只有 Choquet 聚合层，参数量很小。
- rule agent 不需要 GPU。
- LLM API agent 是远程调用，本地不跑模型。
- PyTorch tensor 已通过 `model.device` 和 `to_tensor(..., device)` 放到同一 device，label 也在 `train.py` 中放到同一 device。

已做最小修复：

- `config.py`：`DEVICE = os.getenv("DEVICE", "auto")`
- `main.py`：新增自动检测：
  - `auto` 且 CUDA 可用则 `cuda`
  - 否则 `cpu`
  - 请求 CUDA 但不可用会回退 CPU
- 启动时打印 device。实际验证输出为：
  - `DEVICE requested/effective: auto / cpu`
  - `CUDA available: False`

**2. LLM 是否每个 epoch 调用**
已修复为：LLM 只在预计算阶段调用，训练 epoch 切到 cache-only。

相关改动：

- [src/agents.py](F:/PY_projects/03_STCLABS/Choquet/choquet_agent_vote_demo/src/agents.py)：`LLMBaseAgent` 增加 `cache_only`。cache miss 时：
  - hybrid：回退 rule agent
  - llm：抛错，提示先跑 `scripts/precompute_llm_outputs.py`
- [src/model.py](F:/PY_projects/03_STCLABS/Choquet/choquet_agent_vote_demo/src/model.py)：新增 `set_llm_cache_only()`
- [main.py](F:/PY_projects/03_STCLABS/Choquet/choquet_agent_vote_demo/main.py)：LLM 后端先 `warm_agent_cache(df)`，再 `model.set_llm_cache_only(True)`
- 新增 [scripts/precompute_llm_outputs.py](F:/PY_projects/03_STCLABS/Choquet/choquet_agent_vote_demo/scripts/precompute_llm_outputs.py)

调用次数说明：

- rule 模式：LLM 调用 `0`
- hybrid/llm 小样本 8 条：预计算理论 `8 * 5 = 40` 次；另有 smoke test 最多 5 次
- 1000 条数据、5 agent：理论 `1000 * 5 = 5000` 次，和 epoch 数无关
- 不能每个 epoch 调 LLM：会把成本和耗时乘以 epoch 数，而且远程模型输出可能不稳定

**3. inspired 是否严格离散 Choquet**
不是。当前默认 `inspired` 仍是：

```text
score = sum_i w_i p_i + pair_scale * mean_{i<j}(w_ij * p_i * p_j)
```

它没有：

- agent 分数排序
- 差分 `f_sigma(i)-f_sigma(i-1)`
- 上层集合 `{sigma(i),...,sigma(K)}` 的 capacity 查询
- 完整 capacity 单调性约束

因此只能称为 `Choquet-inspired pairwise aggregation`，不能称为严格离散 Choquet integral。

**4. 新增 discrete_2additive 模式**
新增配置：

```python
CHOQUET_MODE = os.getenv("CHOQUET_MODE", "inspired")
```

允许：

```text
inspired
discrete_2additive
```

实现位置：[src/choquet_layer.py](F:/PY_projects/03_STCLABS/Choquet/choquet_agent_vote_demo/src/choquet_layer.py)

新增模式公式：

```text
C_mu(f) = sum_i [f_sigma(i) - f_sigma(i-1)] * mu({sigma(i), ..., sigma(K)})
```

capacity 用 2-additive Möbius 近似：

```text
mu(S) = sum_{i in S} m_i + sum_{i<j, i,j in S} m_ij
single_m = softmax(raw_single_m)
pair_m = discrete_pair_scale * tanh(raw_pair_m)
mu_normalized(S) = mu(S) / mu(N)
```

它对每个 class 单独排序和聚合，输入 `[batch, agents, classes]`，输出仍是 `[batch, classes]`，兼容现有 train/evaluate。

**5. 单调性**
`discrete_2additive` 目前没有完整强制 capacity 单调性，所以不是数学上完全严格的 Choquet capacity。

已新增 diagnostic：

```python
model.layer.monotonicity_diagnostics()
```

本次小样本训练后输出：

```text
{'checked': 211, 'violations': 0, 'max_violation': 0.0}
```

这只说明该次训练参数未检测到违反，不代表约束被数学强制。

**6. 新增/修改文件**
- `config.py`：新增 `CHOQUET_MODE`，`DEVICE=auto`
- `main.py`：接入 Choquet mode、device auto、LLM cache-only 训练、monotonicity diagnostic
- `src/choquet_layer.py`：保留 inspired，新增 `discrete_2additive`
- `src/model.py`：传入 `choquet_mode`，新增 `set_llm_cache_only`
- `src/agents.py`：新增 LLM cache-only miss 逻辑
- `scripts/precompute_llm_outputs.py`：新增 LLM 输出预计算脚本
- `scripts/test_choquet_formula.py`：新增公式验证脚本
- `README.md`：补充 CPU/GPU、LLM 缓存、Choquet mode 表述

**7. 验证**
通过：

```powershell
python -m py_compile config.py main.py src\choquet_layer.py src\model.py src\train.py src\agents.py src\cache.py scripts\test_choquet_formula.py scripts\precompute_llm_outputs.py
```

通过：

```powershell
python scripts\test_choquet_formula.py
```

结果确认排序：

```text
A4 < A2 < A1 < A3
Diffs: 0.4, 0.2, 0.2, 0.1
Layer score = 0.7400
Expected score = 0.7400
```

通过默认 inspired 小样本训练：

```powershell
$env:CHOQUET_MODE = "inspired"
$env:RUN_SAMPLE_LIMIT = "8"
$env:EPOCHS = "2"
$env:BATCH_SIZE = "2"
python main.py
```

通过新增 discrete_2additive 小样本训练：

```powershell
$env:CHOQUET_MODE = "discrete_2additive"
$env:RUN_SAMPLE_LIMIT = "8"
$env:EPOCHS = "2"
$env:BATCH_SIZE = "2"
python main.py
```

**8. 论文/报告建议表述**
可以说：

- inspired：
  `Choquet-inspired pairwise non-additive aggregation`
- discrete_2additive：
  `a discrete 2-additive Choquet approximation with sorting-difference structure`
- 更严谨：
  `The discrete_2additive mode follows the finite discrete Choquet sorting-difference form, but does not fully enforce capacity monotonicity; therefore it is treated as an approximation.`

不要说：

- `strict Choquet integral` 描述 inspired 模式
- `mathematically exact Choquet integral` 描述当前 discrete_2additive
- `capacity is guaranteed monotone`，因为目前只是 diagnostic，不是约束 乐彩