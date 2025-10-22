import pandas as pd
from city_parse.core import Parse
from typing import List, Dict

# 提取raw_data.example.csv文件的ID列
df = pd.read_csv('raw_data.example.csv')
title_list: List[str] = df['id'].tolist()

title_city_mapping: Dict[str, str] = {}
city_list: List[str] = []

parser = Parse(model_id="qwen3:1.7b")

for title in title_list:
    city_name = parser.parse(title)
    title_city_mapping[title] = city_name
    city_list.append(city_name)


print(title_city_mapping)
print(city_list)

# if you want to save it to a new csv file, run:
df['city'] = df['id'].map(title_city_mapping)
df.to_csv('output.example.csv', index=False)
