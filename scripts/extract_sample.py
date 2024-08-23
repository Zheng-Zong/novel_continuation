import os
import json
import re
import glob
import random
import chardet
from tqdm import tqdm
import tiktoken

categories = [
    "言情", "搞笑", "灵异", "古代", "古风", 
    "散文", "网游", "仙侠", "穿越", "军事", 
    "悬疑", "耽美", "玄幻", "科幻", "TOP"
]


def remove_urls(text):
    url_pattern = r'(?:(?:http|https|ftp|file):\/\/|www\.|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})(?:[^\s]*)'
    text = re.sub(url_pattern, '', text)
    full_width_url_pattern = r'(?:(?:http|https|ftp|file)://|www\.|(?:[a-zA-Z0-9-]+。)+[a-zA-Z]{2,})(?:[^\s]*)'
    text = re.sub(full_width_url_pattern, '', text)
    return text.strip()


def read_novel_file(filepath):
    with open(filepath, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        print(f"读取文件 {filepath} 时编码错误，使用 'utf-8' 重新尝试")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


def extract_samples_from_category(category, sample_count, sample_length):
    category_path = f'./novels/{category}'
    sample_data = []

    novel_files = glob.glob(os.path.join(category_path, '*.txt'))
    random.shuffle(novel_files)

    for novel_file in novel_files:
        content = read_novel_file(novel_file)
        content = content[400:]

        while len(content) > sample_length:
            start_index = random.randint(0, len(content) - sample_length)
            sample_text = content[start_index:start_index + sample_length]

            sample_text = remove_urls(sample_text)

            sample_data.append({
                "text": sample_text.strip()
            })

            if len(sample_data) >= sample_count:
                break
                
        if len(sample_data) >= sample_count:
            break

    return sample_data


def main(target_path, json_path, total_samples, sample_length):
    samples_per_category = total_samples // len(categories)
    all_samples = []

    for category in tqdm(categories, desc="Processing categories"):
        samples = extract_samples_from_category(category, samples_per_category, sample_length)
        all_samples.extend(samples)

    # 计算总token数
    total_tokens = sum(len(tiktoken.encoding_for_model("gpt-4o-mini").encode(sample["text"])) for sample in all_samples)

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_samples, json_file, ensure_ascii=False, indent=4)

    print(f"成功生成 {len(all_samples)} 个样本，保存在 {json_path}")
    print(f"总 token 数: {total_tokens}")


if __name__ == "__main__":
    target_path = '../../novels'
    json_path = '../../sample.json'
    total_samples = 500
    sample_length = 1000

    main(target_path, json_path, total_samples, sample_length)
