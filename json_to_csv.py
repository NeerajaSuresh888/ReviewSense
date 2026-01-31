import json
import pandas as pd

data = []
with open("review_dataset.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 20000:   # stop early
            break
        data.append(json.loads(line))

df = pd.DataFrame(data)
df = df[['reviewText', 'summary', 'overall', 'verified', 'vote']]
df.to_csv("reviews.csv", index=False)