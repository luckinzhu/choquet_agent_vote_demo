# LLM Fallback 错误分析工具使用说明

## 概述

这个脚本用于分析测试预测结果中有多少错误是由 `FALLBACK_NEUTRAL_AFTER_LLM_ERROR` 导致的。

## 运行方式

### 方法1：直接运行（推荐）

在PowerShell中执行：

```powershell
cd F:\PY_projects\03_STCLABS\Choquet\choquet_agent_vote_demo
python scripts\analyze_fallback_errors.py
```

### 方法2：使用auto_run.py集成

如果你想将这个分析步骤加入到自动运行流程中，可以修改 `auto_run.py`：

```python
CONFIG = {
    # ... 其他配置 ...
    "RUN_ANALYZE_FALLBACK": True,  # 添加这一行
}

# 在main函数中添加步骤
if CONFIG.get("RUN_ANALYZE_FALLBACK"):
    steps.append(("Analyze fallback errors", ["scripts/analyze_fallback_errors.py"]))
```

## 脚本功能

### 1. 数据加载
- 读取测试预测CSV文件：`test_predictions.csv`
- 读取LLM缓存JSON文件：`llm_cache_zongxiang.json`

### 2. 分析方法

#### 方法A：检查缓存中的明确标记
搜索缓存中 `raw_text` 字段包含 `"FALLBACK_NEUTRAL_AFTER_LLM_ERROR"` 的条目。

#### 方法B：基于概率模式识别
中性回退（neutral fallback）通常表现为：
- **严格模式**：预测概率接近 [0.5, 0.5]（范围：0.48-0.52）
- **宽松模式**：预测概率在 [0.45, 0.55] 范围内

### 3. 分析内容

脚本会输出：

1. **总体统计**
   - 总预测数
   - 正确/错误数量
   - 错误类型分布（false_positive, false_negative等）

2. **Fallback检测**
   - 缓存中明确的FALLBACK标记数量
   - 基于概率模式的潜在fallback数量
   - 每种错误类型中的fallback比例

3. **详细示例**
   - 展示每个错误类型的典型案例
   - 标记可能的中性回退案例

### 4. 输出文件

分析结果保存在 `outputs/analysis/` 目录：

- **fallback_error_analysis.json**: 完整的分析摘要（JSON格式）
- **strict_neutral_fallback_errors.csv**: 严格模式的潜在fallback错误（概率≈0.5）
- **wider_neutral_fallback_errors.csv**: 宽松模式的潜在fallback错误（0.45≤prob≤0.55）

## 解读结果

### 关键指标

1. **Strict neutral fallback percentage**（严格中性回退比例）
   - > 20%: ⚠️ 高回退率，需要优化
   - 5-20%: ⚡ 中等回退率，建议预计算
   - < 5%: ✅ 低回退率，系统运行良好

2. **Explicit fallback in cache**（缓存中的明确fallback标记）
   - 这个数字表示实际发生LLM错误并使用中性回退的次数

### 优化建议

如果检测到高回退率：

1. **提高LLM API稳定性**
   ```powershell
   # 增加超时时间
   $env:LLM_TIMEOUT="120"
   ```

2. **使用混合模式**
   ```powershell
   # 改用hybrid模式，LLM失败时回退到rule-based
   $env:AGENT_BACKEND="hybrid"
   ```

3. **预计算LLM输出**
   ```powershell
   # 训练前预先计算所有LLM输出
   python scripts/precompute_llm_outputs.py
   ```

4. **检查API配置**
   - 确认API密钥有效
   - 检查网络连接
   - 验证LLM网关地址

## 示例输出

```
================================================================================
LLM Fallback Error Analysis Tool
================================================================================
Loading predictions from: ...\test_predictions.csv
Total predictions: 639
Loading cache from: ...\llm_cache_zongxiang.json
Total cache entries: 1493

================================================================================
Analyzing Fallback Errors
================================================================================

Total error cases: 127

Error type distribution:
  false_positive: 68
  false_negative: 59

--------------------------------------------------------------------------------
Checking for FALLBACK_NEUTRAL_AFTER_LLM_ERROR in cache...
--------------------------------------------------------------------------------

Found 23 cache entries with FALLBACK_NEUTRAL_AFTER_LLM_ERROR

--------------------------------------------------------------------------------
Analyzing prediction patterns for neutral fallback indicators...
--------------------------------------------------------------------------------

Potential neutral fallback errors (prob ≈ 0.5): 15
Wider range potential fallbacks (0.45 ≤ prob ≤ 0.55): 32

================================================================================
Summary Statistics
================================================================================

Total predictions: 639
Correct predictions: 512
Total errors: 127

Errors with strict neutral fallback pattern (≈0.5): 15 (11.81%)
Errors with wider neutral pattern (0.45-0.55): 32 (25.20%)
Cache entries with explicit FALLBACK_NEUTRAL marker: 23

⚡ Moderate fallback rate detected.
   Consider pre-computing LLM outputs for more stable training.
```

## 故障排除

### 问题1：找不到文件

确保路径正确：
```python
# 在脚本中修改这些路径
predictions_csv = project_root / "outputs" / "runs" / "YOUR_RUN_FOLDER" / "test_predictions.csv"
cache_json = project_root / "outputs" / "llm_cache_zongxiang.json"
```

### 问题2：pandas未安装

```powershell
pip install pandas
```

### 问题3：编码错误

脚本已使用 `encoding='utf-8-sig'` 保存CSV，支持中文。

## 自定义分析

如果需要调整检测阈值，修改脚本中的这些参数：

```python
# 严格模式阈值
potential_fallbacks = error_cases[
    (error_cases['pred_prob_0'] >= 0.48) &  # 修改这里
    (error_cases['pred_prob_0'] <= 0.52) &  # 修改这里
    ...
]

# 宽松模式阈值
wider_fallbacks = error_cases[
    (error_cases['pred_prob_0'] >= 0.45) &  # 修改这里
    (error_cases['pred_prob_0'] <= 0.55) &  # 修改这里
    ...
]
```

## 相关文件

- 脚本位置：`scripts/analyze_fallback_errors.py`
- 输入文件：
  - `outputs/runs/*/test_predictions.csv`
  - `outputs/llm_cache_zongxiang.json`
- 输出文件：`outputs/analysis/*`

## 注意事项

1. **缓存文件可能很大**：首次加载可能需要几秒
2. **概率阈值需要根据实际情况调整**：不同模型的回退模式可能略有不同
3. **中性回退不是唯一的错误来源**：还要考虑模型本身的判断错误
4. **定期清理分析结果**：避免outputs/analysis目录积累过多文件
