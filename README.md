# 🌟 续梦笔

🎉 **项目描述**  
续梦笔是一款基于大语言模型 (LLM) 的工具，旨在对现有作品进行续写。

## 📦 安装过程

我们推荐使用 [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 进行环境管理。以下是安装步骤：

### 🐧 对于 Linux 用户

在项目路径下运行以下命令以创建虚拟环境：

```bash
conda env create -f linux_environment.yml
```

### 🖥️ 对于 Windows 用户

在项目路径下运行以下命令以创建虚拟环境：

```bash
conda env create -f windows_environment.yml
```

## 🔧 使用 FastChat

如果您想使用本地语言大模型（例如Yuan2.0），可以使用 [FastChat](https://github.com/lm-sys/FastChat) 搭建与 OpenAI 兼容的 API。请按照以下步骤进行设置：

1. 克隆 FastChat 仓库并安装依赖。
2. 按照 FastChat 仓库中的[说明](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)搭建 API 接口。
3. 修改本项目中的 `config.yaml` 文件中的 `llm_model` 和 `llm_api_base` 设置。
4. 嵌入模型同理。

## 🚀 开始使用

在成功安装后，您可以通过以下命令启动续梦笔：

```bash
streamlit run app.py
```
---
## 🚧 正在施工中