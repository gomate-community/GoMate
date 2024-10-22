import requests
import json
import pandas  as pd
url='http://10.208.61.117:9200/document_share_data_30_news/_search?q=%E7%89%B9%E6%9C%97%E6%99%AE&size=3000&sort=publish_time:desc'

response=requests.get(url)


with open('data.json','w',encoding='utf-8') as f:
    json.dump(response.json(),f,ensure_ascii=False,indent=4)


with open('data.json','r',encoding='utf-8') as f:
    data=json.load(f)


sources=[hit['_source'] for hit in data['hits']['hits']]
print(len(sources))



source_df=pd.DataFrame(sources)
print(source_df)
print(source_df.columns)
print(source_df['paragraph_ids'])

source_df['id']=source_df.index
source_df['id']='source_'+source_df['id'].astype(str)

print(source_df)

source_df[['id','title','content']].to_excel('data.xlsx')