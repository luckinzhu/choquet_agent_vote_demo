# Few-shot功能实现总结

## 修改概览

本次更新为LLM智能体添加了few-shot示例支持，使每个专家Agent能够基于示例更好地理解判断标准。

## 修改的文件

### 1. config.py
**新增配置项**：
- `FEWSHOT_ENABLED`: 控制是否启用few-shot功能（默认true）
- `FEWSHOT_DIR`: few-shot示例文件目录路径

### 2. src/agents.py
**核心修改**：

#### LLMBaseAgent类
- 新增类属性 `fewshot_file = None`
- 新增实例属性 `self.fewshot_examples`
- 新增方法 `_load_fewshot_examples()`: 从JSON文件加载示例
- 重构方法 `user_prompt()`: 在prompt中插入few-shot示例

#### 各个LLM智能体类
为以下5个智能体类添加了 `fewshot_file` 属性：
- `LLMSemanticAgent` → "semantic.json"
- `LLMEmotionAgent` → "emotion.json"
- `LLMIntentionAgent` → "intention.json"
- `LLMLexicalAgent` → "lexical.json"
- `LLMConsistencyAgent` → "consistency.json"

### 3. auto_run.py
**新增配置**：
- `FEWSHOT_ENABLED`: 添加到CONFIG字典和ENV_KEYS列表
- `RUN_TEST_FEWSHOT`: 控制是否运行few-shot测试（默认True）
- 在 `print_config()` 中显示FEWSHOT_ENABLED状态
- 在 `main()` 中添加few-shot测试步骤

### 4. 新增文件

#### data/fewshot_examples/*.json (5个文件)
为每个智能体创建了包含3个示例的JSON文件：
- semantic.json: 语义上下文判断示例
- emotion.json: 情绪态度判断示例
- intention.json: 点击诱导意图判断示例
- lexical.json: 词汇表层模式判断示例
- consistency.json: 一致性/矛盾判断示例

#### scripts/test_fewshot.py
完整的测试脚本，包含4个测试用例：
1. Few-shot配置加载测试
2. Few-shot示例加载测试
3. Prompt生成测试（验证示例是否正确插入）
4. AgentFactory集成测试

#### plans/FEWSHOT_GUIDE.md
详细的使用指南文档

## 使用方法

### 快速测试
```powershell
# 运行auto_run.py会自动执行few-shot测试
python auto_run.py
```

### 单独测试few-shot功能
```powershell
python scripts/test_fewshot.py
```

### 禁用few-shot
```powershell
$env:FEWSHOT_ENABLED="false"
python main.py
```

## 技术亮点

1. **模块化设计**：每个智能体独立管理自己的few-shot文件
2. **灵活开关**：通过环境变量快速启用/禁用
3. **容错处理**：文件不存在或格式错误时优雅降级
4. **缓存兼容**：与现有LLM缓存机制完全兼容
5. **易于扩展**：添加新智能体只需指定fewshot_file属性

## 验证清单

✅ 配置文件正确添加
✅ 5个few-shot示例文件创建
✅ LLMBaseAgent支持few-shot加载
✅ 所有LLM智能体指定了对应的fewshot_file
✅ user_prompt正确插入few-shot示例
✅ auto_run.py集成测试步骤
✅ 测试脚本无语法错误
✅ 使用指南文档完成

## 下一步建议

1. 运行 `python auto_run.py` 验证整体流程
2. 根据实际效果调整few-shot示例
3. 考虑为不同任务创建不同的few-shot集合
4. 监控LLM输出质量变化

## 注意事项

- Few-shot仅在 `AGENT_BACKEND="llm"` 或 `"hybrid"` 时生效
- Rule-based agents不受此功能影响
- 首次运行会加载示例并可能调用LLM API（如未缓存）
- 建议先运行测试脚本确认功能正常
