import pandas as pd
df = pd.read_csv("movies.csv")
df = df.drop("title", 1)
df.to_csv("mov.csv", index=False)

