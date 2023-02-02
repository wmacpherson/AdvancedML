import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# sec_reports = load_dataset("JanosAudran/financial-reports-sec", 'small_full')
us_equity_news=pd.read_csv("/Users/williammacpherson/Documents/FinancialDatasets/us_equities_news_dataset.csv", nrows=100)
us_equity_news.dropna()
us_equity_news=us_equity_news.drop(['provider','url','article_id','id','category'], axis=1)
us_equity_news=us_equity_news.rename(columns={'release_date':'date','ticker':'stock','title':'headline'})

analyst_ratings=pd.read_csv("/Users/williammacpherson/Documents/FinancialDatasets/Daily Financial News/raw_analyst_ratings.csv")
analyst_ratings.dropna()
analyst_ratings=analyst_ratings.drop(['Unnamed: 0','url','publisher'], axis=1)

df_array = np.array(us_equity_news)
df_list = list(df_array[:,2])

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
#tokenize text to be sent to model
inputs = tokenizer(df_list, padding = True, truncation = True, return_tensors='pt')
outputs = model(**inputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

model.config.id2label


positive = predictions[:, 0].tolist()
negative = predictions[:, 1].tolist()
neutral = predictions[:, 2].tolist()

table = {'Headline':df_list,
         "Positive":positive,
         "Negative":negative, 
         "Neutral":neutral}
      
df2 = pd.DataFrame(table, columns = ["Headline", "Positive", "Negative", "Neutral"])

print(df2)