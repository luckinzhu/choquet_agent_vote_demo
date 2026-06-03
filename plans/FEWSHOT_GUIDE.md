# Few-shot功能使用指南

## 概述

本项目已支持为每个LLM智能体配置独立的few-shot示例，帮助LLM更好地理解各专家视角的判断标准。

## 快速开始

### 1. 启用Few-shot功能

在 `auto_run.py` 中设置：

```python
CONFIG = {
    "AGENT_BACKEND": "llm",  # 或 "hybrid"
    "FEWSHOT_ENABLED": "true",  # 启用few-shot
    "RUN_TEST_FEWSHOT": True,  # 运行few-shot测试
    # ... 其他配置
}
```

或通过环境变量：

```powershell
$env:FEWSHOT_ENABLED="true"
$env:AGENT_BACKEND="llm"
python auto_run.py
```

### 2. 运行测试

```powershell
# 方式1：使用auto_run.py（推荐）
python auto_run.py

# 方式2：直接运行测试脚本
python scripts/test_fewshot.py
```

测试将验证：
- ✓ Few-shot配置文件加载
- ✓ 每个智能体的示例正确加载
- ✓ Prompt中包含few-shot示例
- ✓ AgentFactory集成正常

### 3. 查看Few-shot文件位置

所有few-shot示例文件位于：
```
data/fewshot_examples/
├── semantic.json        # 语义上下文专家
├── emotion.json         # 情绪态度专家
├── intention.json       # 点击诱导意图专家
├── lexical.json         # 词汇表层模式专家
└── consistency.json     # 一致性/矛盾专家
```

## 自定义Few-shot示例

### 文件格式

每个JSON文件包含一个示例数组，每个示例结构如下：

```json
[
  {
    "example": "这里是示例文本内容",
    "class_0_probability": 0.85,
    "class_1_probability": 0.15,
    "confidence": 0.9,
    "explanation": "这是判断的中文解释"
  }
]
```

### 字段说明

- **example**: 示例文本（标题或标题+内容）
- **class_0_probability**: 非标题党的概率 (0-1)
- **class_1_probability**: 标题党的概率 (0-1)
- **confidence**: 置信度 (0-1)
- **explanation**: 判断理由（建议不超过50字）

### 编辑示例

以Semantic Agent为例：

1. 打开 `data/fewshot_examples/semantic.json`
2. 添加或修改示例
3. 保存文件
4. 重新运行测试或训练

**注意**：
- 建议每个智能体提供2-5个高质量示例
- 示例应体现该智能体的专业视角
- 覆盖正例和负例（标题党和非标题党）

## 关闭Few-shot功能

如需暂时禁用few-shot：

```python
# 在auto_run.py中
"FEWSHOT_ENABLED": "false",
```

或：

```powershell
$env:FEWSHOT_ENABLED="false"
```

## 工作原理

1. **加载阶段**：每个LLM智能体初始化时自动加载对应的few-shot文件
2. **Prompt构建**：在调用LLM时，few-shot示例会被插入到prompt中
3. **缓存机制**：LLM的输出会被缓存，避免重复API调用
4. **灵活开关**：可通过环境变量快速启用/禁用

## 故障排查

### 问题1：Few-shot示例未加载

检查：
- `FEWSHOT_ENABLED` 是否设置为 `"true"`
- JSON文件是否存在于 `data/fewshot_examples/` 目录
- JSON格式是否正确（可使用JSON验证工具）

### 问题2：测试失败

运行详细测试：
```powershell
python scripts/test_fewshot.py
```

查看具体哪个步骤失败。

### 问题3：LLM输出不符合预期

- 检查few-shot示例质量
- 确保示例与智能体的专业视角一致
- 尝试调整示例数量（2-5个为宜）

## 最佳实践

1. **示例质量 > 数量**：2-3个高质量示例优于多个低质量示例
2. **覆盖边界情况**：包含容易混淆的案例
3. **保持一致性**：同一智能体的示例应遵循相同的判断标准
4. **定期更新**：根据实际效果调整和优化示例
5. **分别优化**：每个智能体独立优化其few-shot示例

## 技术细节

### 相关文件

- `config.py`: FEWSHOT_ENABLED, FEWSHOT_DIR 配置
- `src/agents.py`: LLMBaseAgent._load_fewshot_examples(), user_prompt()
- `auto_run.py`: RUN_TEST_FEWSHOT 开关
- `scripts/test_fewshot.py`: 测试脚本

### 代码架构

```
LLMBaseAgent (基类)
├── fewshot_file: 指定JSON文件名
├── _load_fewshot_examples(): 加载示例
└── user_prompt(): 构建包含few-shot的prompt

各个LLM智能体继承并指定fewshot_file:
├── LLMSemanticAgent → semantic.json
├── LLMEmotionAgent → emotion.json
├── LLMIntentionAgent → intention.json
├── LLMLexicalAgent → lexical.json
└── LLMConsistencyAgent → consistency.json
```

## 示例效果对比

### 不使用Few-shot
```
任务描述：判断是否为标题党
标签定义：...
待判断文本：震惊！这个结果让人难以置信！
请从你的专家视角输出严格 JSON：...
```

### 使用Few-shot
```
任务描述：判断是否为标题党
标签定义：...

以下是一些示例，供你参考判断标准：

示例 1：
文本：震惊！这个结果让人难以置信！
输出：{
  "class_0_probability": 0.05,
  "class_1_probability": 0.95,
  "confidence": 0.92,
  "explanation": "强烈的情绪词汇制造感官刺激，典型的标题党手法"
}

请按照上述示例的判断标准，对待判断文本进行分析。

待判断文本：...
```

通过提供明确的判断标准，LLM能更准确地模拟各专家视角。
