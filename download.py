# python文件命名 download_model.py
from modelscope import snapshot_download
#指定下载目录
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct',cache_dir='/root/autodl-tmp')
print(f"模型已下载到:{model_dir}")
