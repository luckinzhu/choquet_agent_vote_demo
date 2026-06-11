# API 配置快速切换指南

## 📋 概述

现在您可以通过修改 `auto_run.py` 中的单个参数来快速切换不同的 API 配置，无需手动编辑 `.env` 文件。

## 🚀 使用方法

### 1. 打开 `auto_run.py` 文件

找到第 **33-52** 行的 API 配置预设部分：

```python
# Option 1: DeepSeek API (current)
API_PRESET = "deepseek"
API_CONFIGS = {
    "deepseek": {
        "LLM_BASE_URL": "https://api.deepseek.com/v1",
        "LLM_API_KEY_VALUE": "",  # Read from .env file
        "LLM_MODEL": "deepseek-v4-flash",
    },
    "gpt55": {
        "LLM_BASE_URL": "https://xiaohumini.site/v1",
        "LLM_API_KEY_VALUE": "sk-qFEi64pRVOi7gTrP3pSzsc1ROjeq8CnERcsO9xJBZtlEV3EW",
        "LLM_MODEL": "gemini-3.1-flash-lite",
    },
}
```

### 2. 切换 API 配置

只需修改 `API_PRESET` 的值：

#### 使用 DeepSeek API（从 .env 读取密钥）
```python
API_PRESET = "deepseek"
```
将自动配置：
- `LLM_BASE_URL`: https://api.deepseek.com/v1
- `LLM_MODEL`: deepseek-v4-flash
- `LLM_API_KEY`: 从 .env 文件读取

#### 使用小虎 API（硬编码密钥）
```python
API_PRESET = "gpt55"
```
将自动配置：
- `LLM_BASE_URL`: https://xiaohumini.site/v1
- `LLM_MODEL`: gemini-3.1-flash-lite
- `LLM_API_KEY`: sk-qFEi64pRVOi7gTrP3pSzsc1ROjeq8CnERcsO9xJBZtlEV3EW

### 3. 运行程序

```bash
python auto_run.py
```

您会看到输出中显示：
```
Using API preset: deepseek
```

## ➕ 添加新的 API 配置

如果您需要添加新的 API 提供商，只需在 `API_CONFIGS` 字典中添加新条目：

```python
API_CONFIGS = {
    "deepseek": {
        "LLM_BASE_URL": "https://api.deepseek.com/v1",
        "LLM_API_KEY_VALUE": "",
        "LLM_MODEL": "deepseek-v4-flash",
    },
    "gpt55": {
        "LLM_BASE_URL": "https://xiaohumini.site/v1",
        "LLM_API_KEY_VALUE": "sk-qFEi64pRVOi7gTrP3pSzsc1ROjeq8CnERcsO9xJBZtlEV3EW",
        "LLM_MODEL": "gemini-3.1-flash-lite",
    },
    # 添加新的 API 配置
    "openai": {
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY_VALUE": "sk-your-openai-key-here",
        "LLM_MODEL": "gpt-4o",
    },
    "custom_api": {
        "LLM_BASE_URL": "https://your-custom-api.com/v1",
        "LLM_API_KEY_VALUE": "your-api-key",
        "LLM_MODEL": "your-model-name",
    },
}
```

然后切换：
```python
API_PRESET = "openai"  # 或 "custom_api"
```

## 💡 配置说明

### 方式一：从 .env 读取密钥（推荐）
```python
"deepseek": {
    "LLM_BASE_URL": "https://api.deepseek.com/v1",
    "LLM_API_KEY_VALUE": "",  # 空字符串表示从 .env 读取
    "LLM_MODEL": "deepseek-v4-flash",
}
```

### 方式二：硬编码密钥
```python
"gpt55": {
    "LLM_BASE_URL": "https://xiaohumini.site/v1",
    "LLM_API_KEY_VALUE": "sk-qFEi64pRVOi7gTrP3pSzsc1ROjeq8CnERcsO9xJBZtlEV3EW",
    "LLM_MODEL": "gemini-3.1-flash-lite",
}
```

## ⚠️ 注意事项

1. **安全性**：如果使用硬编码密钥，请不要将包含真实密钥的文件提交到版本控制系统
2. **优先级**：`API_PRESET` 的配置会覆盖 CONFIG 中的默认值
3. **验证**：每次切换后，运行时会显示当前使用的 preset 名称

## 🎯 优势

✅ 无需编辑 `.env` 文件  
✅ 一键切换 API 配置  
✅ 支持多个预设配置  
✅ 清晰的配置管理  
✅ 易于扩展新的 API 提供商
