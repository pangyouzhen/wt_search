import pandas as pd

df = pd.read_csv("./data/new.train.query.txt",sep="\t",names=["query"])

df2 = pd.read_csv("./data/qrels.train.tsv",sep="\t",names=["query","doc"])

df3 = pd.read_csv("./data/corpus.tsv",sep="\t",names=["doc"])

d1 = df.squeeze().to_dict()
d3 = df3.squeeze().to_dict()

df2["query"] = df2["query"].map(d1)

df2.columns

df2["doc"] = df2["doc"].map(d3)

df2.to_csv("./data/train.csv",index=False)


