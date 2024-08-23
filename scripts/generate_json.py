import os
import json
import re
import glob
import chardet
from tqdm import tqdm

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
    return text


def read_novel_file(filepath):
    with open(filepath, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    with open(filepath, 'r', encoding=encoding) as f:
        return f.read()


def extract_samples_from_category(category, sample_count, input_length, output_length):
    category_path = f'./novels/{category}'
    sample_data = []

    novel_files = glob.glob(os.path.join(category_path, '*.txt'))
    
    for novel_file in novel_files:
        content = read_novel_file(novel_file)
        
        content = content[400:]

        while len(content) > input_length + output_length:
            input_text = content[:input_length]
            output_text = content[input_length:input_length + output_length]
            if len(output_text) < output_length:
                break
            
            input_text = remove_urls(input_text)
            output_text = remove_urls(output_text)

            sample_data.append({
                "instruction": f"请根据以下内容续写风格为{category}小说",
                "input": input_text.strip(),
                "output": output_text.strip()
            })

            content = content[input_length:]

            if len(sample_data) >= sample_count:
                break
                
        if len(sample_data) >= sample_count:
            break

    return sample_data


def main(target_path, json_path, total_samples, input_length, output_length):
    samples_per_category = total_samples // len(categories)
    all_samples = []

    for category in tqdm(categories, desc="Processing categories"):
        samples = extract_samples_from_category(category, samples_per_category, input_length, output_length)
        all_samples.extend(samples)

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_samples, json_file, ensure_ascii=False, indent=4)

    print(f"成功生成 {len(all_samples)} 个样本，保存在 {json_path}")


if __name__ == "__main__":
    target_path = '../../novels'
    json_path = '../../small_novels_dataset.json'
    total_samples = 2000
    input_length = 400
    output_length = 1600

    main(target_path, json_path, total_samples, input_length, output_length)
