import os
# 关键修正：必须在 import datasets 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from datasets import load_dataset

target_dir = "/root/autodl-tmp/LLaMA-Factory/data"
os.makedirs(target_dir, exist_ok=True)

print("正在从国内镜像下载 ChnSentiCorp 数据集...")
cache_path = os.path.join(target_dir, ".hf_cache")
dataset = load_dataset("lansinuote/ChnSentiCorp", cache_dir=cache_path)

print(f"【查看】原始数据集已成功缓存至: {os.path.abspath(cache_path)}")

sft_data = []
for item in dataset['train']:
    label_text = "正向" if item['label'] == 1 else "负向"
    sft_data.append({
        "instruction": "你是一个文本情感分析专家。请阅读用户的评论，判断其情感倾向是【正向】还是【负向】。只需回答这两个词之一，不要说任何多余的话。",
        "input": item['text'],
        "output": label_text
    })

output_path = os.path.join(target_dir, "chn_senti_sft.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=2)

print("\n" + "="*50)
print(f"【成功】大模型微调数据集已生成！")
print(f"【位置】已保存至: {output_path}")
print(f"【规模】总计 {len(sft_data)} 条数据")
print("="*50)
print("数据样例预览：\n", json.dumps(sft_data[0], ensure_ascii=False, indent=2))