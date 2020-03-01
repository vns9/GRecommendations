import pandas as pd
df = pd.read_csv("ratings.csv")
df['rating'].replace(df['rating']*10)
df.to_csv(index=False)
