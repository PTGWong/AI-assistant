# Gradio-based AI Assistant GUI

一个基于Gradio构建的AI助手图形界面，支持对接自定义API服务。通过环境变量配置API参数，轻松实现对话交互。

## ✨ 功能特性
- **流式响应**：实时显示AI生成内容
- **历史记录保存**：自动保存对话上下文
- **系统提示词配置**：支持预设系统角色设定

## 🛠️ 环境要求
- Python 3.8+
- pip 20.0+

## ⚙️ 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/PTGWong/AI-assistant.git
cd AI-assistant
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 🔑 配置说明

在项目根目录创建.env文件：

```env
# OpenAI兼容API配置
openai_api_key = your-api-key-here
openai_api_base = https://your-api-endpoint.com/v1
```
## 🚀 启动应用

```bash
python assistant.py
```
访问 http://localhost:7860 开始使用
