## LoRA部分核心讲解
![[file-20260529150408154.png]]

微调当中，是通过反向传播来更新参数的，因此左边的冻结矩阵不参与更新

## LoRA模板在大模型的位置
![[file-20260529150744181.png]]



![[file-20260529151329127.png]]



多头注意力机制的核心，就是四个线形投影矩阵，W_Q、W_K、W_V、W_O矩阵，分别对应query key val 以及输出投影

实际证明了，只要改变W_Q、W_V就能显著改变模型的注意力分布

![[file-20260529151642913.png]]

W_up层：把特征从低维映射到高维
W_gate层：控制信息通过的比例
W_down层：把高维的空间数据，映射到原来的维度

任务设计到负责任务 、代码推理的时候，微调MLP层非常关键

![[file-20260529152042149.png]]


面试经典



![[file-20260529162808851.png]]
LoRA微调当中 A、B矩阵需要满足两个条件
1.原始的A $\cdot$B是0 从而确保原本的模型的能力不会被干扰

2.确保我们在训练过程当中的有效性：
训练过程当中，梯度反向传播一定要能更新到这两个模块
因此损失函数对A、B这两个的矩阵梯度不同时为0

如果A、B矩阵全为0，损失函数对A、B的梯度为0，两个梯度都为0，训练过程中不会更新，无意义

A，B都不为0，A，B乘积不为0，初始时会给原先大模型的加上噪声

一个为0，一个初始化不为0


## LoRA微调实际指南

conda init 然后关闭终端（点击垃圾桶键盘）

看到有base，说明启动成功了

同时租赁云服务器的话，一定要在数据盘下下载，
确定当前路径，然后再打开

```python

pwd #显示当前文件目录

cd 数据盘路径

ls #展示目录

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git



#autodl自带的加速方法
source /etc/network_tjrbo


git clone --depth 1 https://gh-proxy.com/https://github.com/hiyouga/LLaMA-Factory.git #实测可以




cd llamafactory 

pip install -e .[metrics] #安装llamafactory的依赖

```


然后的话，就通过vscode file 打开文件夹 llamafactory

模型的话，需要下载到本地，在本地新建模型运行文件
```python 
# python文件命名 download_model.py
from modelscope import snapshot_download
#指定下载目录
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct',cache_dir='/root/autodl-tmp')
print(f"模型已下载到:{model_dir}")
```

```python

pip install bitsandbytes>=0.39.0 #下载QloRA所需的库


```
数据下载到llmfactory当中的data文件夹下了，但是还需要注册数据，打开dataset_info.json文件，复制对应文件的信息

```python

"medical_1k":{
"file_name":"medical_1k.json",
"formatting":"sharegpt",
"columns":{
"messages":"conversations"
},
tags":{
  "role_tag":"from",
  "content_tag":"value",
  "user_tag":"human",
  "assistant_tag":"gpt"
  }
}
```


使用llamafactory进行 LoRA微调的话，需要事先打开llamafactory的文件

```python

llamafactory-cli webui #打开llamafactory webui

pwd #输出当前路径


```

遇到Pytorch不匹配的问题

```python
OSError: libcudart.so.13: cannot open shared object file: No such file or directory
```

这种时候，先检查Pytorch的版本，如果不相等，说明二者不匹配



```python

# 系统支持的 CUDA 版本（驱动支持的最高版本）
nvidia-smi | grep "CUDA Version"

# PyTorch 编译时捆绑的 CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 如果系统 CUDA 是 11.8，重装pytroch代码如下所示：
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 后面118就代表了118版本

```


lora缩放系数一般是lora秩的两倍




微调结束之后，我们可以在llamafactory这个saves目录之下，找到我们这一次微调过程当中的记录
train_loss.png 就是微调过程当中的损失的曲线变化过程

all_results.json记录了训练过程当中的一些详细信息

### 测试微调效果

1.打开llamafactor页面，从原先的train选择到Chats

2.检查点路径，选择微调的那个路径（会显示train_具体时间）

3.选择“加载模型”

4.加载成功之后，就可以问模型一些问题


如果想对比测试前后的话，可以将选择点路径定为空



## 面试问题总结

sft到什么程度，你会考虑做rl

1.模型能力维度：
SFT训练到位的标志是，模型已经具备了稳定的指令跟随能力，问问题，能老师回答，不会再去做文本补全 or 答非所问，即模型的输出分布，已经基本对齐语言模型的分布了 

2.从数据规模来看 
deepseek V2使用了150万条高质量数据，120W有益数据，30W安全数据，训练了2个epoch，学习率控制在5$\cdot 10^{-6}$ 来保证收敛

3.reward采样分布

使用sft的模型对同一个prompt采样多个回答时，如果reward分布相对均匀，有高分、低分，说明模型已经会回答了，只是质量参差不弃。这个时候RL才有用武之地，如果采样出来的回答reward都很低，说明sft没训练到位

4.标准成本
sft需要人工标注完整的（prompt，response）对，成本相对较高，
RL阶段可以让模型自己生成候选答案，人只需要做好标注 or 排序，甚至可以用reward model 来辅助，标注效率大大提升




