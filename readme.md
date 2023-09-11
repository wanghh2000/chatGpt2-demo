# ChatGPT-2 Demo

基于 chatGpt-2实现与Ai对话功能。  
在本地部署ChatGPT-2模型，并实现一个简单的对话客户端（基于flask）。   


## 环境准备

确保您已安装以下依赖：

- Python 3.6 或更高版本
- pip

## 安装

1. 克隆项目：

```shell
 git clone https://github.com/jianwi/chatGpt2-demo.git
```
2. 安装所需依赖：
```shell
pip install -r requirements.txt
```

## 使用方法

1. 运行Flask API：
```shell
python app.py
```

默认情况下，API将在`http://127.0.0.1:5000/`上运行。

在浏览器中打开`http://127.0.0.1:5000/`，您将看到一个简单的前端页面，可以输入问题并获取GPT-2生成的回复。

## 注意事项

- 可以根据需要调整GPT-2模型的参数，以获得更高质量的生成结果。
比如增加max_length的值，可以输出更多字符。
- 如果运行不起来，可能需要安装 PyTorch库:
```shell
pip install torch
```

```
# app.py
conda create -n gpt2 python=3.8
conda activate gpt2
pip install -r requirements.txt
pip install torch
python app.py
```

```
# chatglm.py
conda create -n chatglm python=3.8
conda activate chatglm

pip install transformers sentencepiece cpm_kernels

# torch on gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# pytorch on cpu
pip install torch torchvision torchaudio  

python chatglm.py
```

https://github.com/THUDM/ChatGLM2-6B

```
#chatglm2-6b-int4.py

conda create -n chatgpt2 python=3.8
conda activate chatgpt2

pip install protobuf
pip install transformers==4.30.2
pip install cpm_kernels
pip install torch>=2.0
pip install gradio
pip install mdtex2html
pip install sentencepiece
pip install accelerate
pip install sse-starlette
pip install streamlit>=1.24.0

python chatglm2-6b-int4.py
```

## 演示
![问题1](./demo/1.png)

在html中显示hello world
![问题2](./demo/2.png)

调节max_length值，输出更多字符
![问题3](./demo/3.png)
