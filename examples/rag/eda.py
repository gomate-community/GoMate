import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import jieba

# 读取数据
news = []
with open('../../data/competitions/xunfei/corpus.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        news.append(line.strip())

df = pd.DataFrame({'news': news})

# 1. 查看数据的基本信息
print("数据的基本信息：")
print(df.info())
print("\n数据的统计摘要：")
print(df.describe())

# 2. 新闻长度分析
df['news_length'] = df['news'].apply(len)

# 可视化新闻长度分布
plt.figure(figsize=(10, 6))
plt.hist(df['news_length'], bins=50, color='skyblue')
plt.title('new length')
plt.xlabel('length of chars')
plt.ylabel('nums of news')
plt.show()

# 3. 词频分析
# 使用jieba分词
df['news_words'] = df['news'].apply(lambda x: jieba.lcut(x))

# 统计词频
all_words = [word for words in df['news_words'] for word in words if len(word.strip())>2]
word_counts = Counter(all_words)

# 查看最常见的10个词
common_words = word_counts.most_common(20)
print("\n新闻中最常见的10个词：")
for word, count in common_words:
    print(f"{word}: {count} 次")

# 4. 假设新闻来源是含有"记者"等关键词的新闻行
df['source'] = df['news'].apply(lambda x: '记者' in x or '本报讯' in x)

# 查看有新闻来源的条数
print("\n含有新闻来源的条数：", df['source'].sum())
