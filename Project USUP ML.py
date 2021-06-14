# The libraries needed

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# DAE
Some cleaning

df_ks = pd.read_csv("ks-projects-201801.csv")

df_ks.drop(columns="ID", inplace=True)

df_ks.deadline = pd.to_datetime(df_ks.deadline)
df_ks.launched = pd.to_datetime(df_ks.launched)

df_ks = df_ks.dropna()

# PCA

scaler = StandardScaler()
pca = PCA()

df_ks_scale = scaler.fit_transform(df_ks.select_dtypes("number"))

pca_ks=pca.fit_transform(df_ks_scale)
print(pca.explained_variance_ratio_)

pc1_ks=[]
pc2_ks=[]
pc3_ks=[]
pc4_ks=[]
pc5_ks=[]
pc6_ks=[]
for i in pca_ks:
    pc1_ks.append(i[0])
    pc2_ks.append(i[1])
    pc3_ks.append(i[2])
    pc4_ks.append(i[3])
    pc5_ks.append(i[4])
    pc6_ks.append(i[5])

# Adding the PCA components to the DataFrame
df_ks["pc1"]=pc1_ks
df_ks["pc2"]=pc2_ks
df_ks["pc3"]=pc3_ks
df_ks["pc4"]=pc4_ks
df_ks["pc5"]=pc5_ks
df_ks["pc6"]=pc6_ks

plt.figure(figsize=(14, 10))
sns.scatterplot(data=df_ks, x="pc1", y="pc2")
plt.title("PCA plot with the first 2 components (ca 90 % variance)")
plt.show()

# Using an elbow plot to figure out the optimal number of clusters, in this case 3

elbow_viz = KElbowVisualizer(KMeans(), k=(1, 8), timings=False)
elbow_viz.fit(df_ks_scale)
plt.figure(figsize=(12, 8))
plt.title("Elbow plot")
plt.show()

# 3 clusters

kmeans = KMeans(n_clusters = 3)
kmeans.fit(df_ks_scale)

df_ks["clusters"] = kmeans.labels_ 

# Adding the cluster labels to both the scaled and unscaled DF

df_ks["clusters"] = kmeans.labels_
df_ks_scale = pd.DataFrame(df_ks_scale, columns=['goal', 'pledged', 'backers', 'usd pledged', 'usd_pledged_real',
       'usd_goal_real'])

df_ks_scale["clusters"] = kmeans.labels_

df_ks.clusters.value_counts()

# renaming the clusters to something more exciting

df_ks.clusters.loc[df_ks.clusters == 0] = "The not-so-interesting"
df_ks.clusters.loc[df_ks.clusters == 2] = "High Profile"
df_ks.clusters.loc[df_ks.clusters == 1] = "The Overly Ambitious"
df_ks_scale.clusters.loc[df_ks_scale.clusters == 0] = "The not-so-interesting"
df_ks_scale.clusters.loc[df_ks_scale.clusters == 2] = "High Profile"
df_ks_scale.clusters.loc[df_ks_scale.clusters == 1] = "The Overly Ambitious"

## EDA
Trying to see what makes the cluster differ from each other

df_ks.groupby("clusters").mean()

df_ks_scale.groupby("clusters").mean()

plt.figure(figsize=(14, 10))
sns.scatterplot(data=df_ks, x="pc1", y="pc2", hue="clusters", style="state")
plt.show()

plt.figure(figsize=(14, 10))
sns.scatterplot(data=df_ks, x="pledged", y="goal", hue="clusters", style="state")
plt.show()

# Making seperate DataFrames for each cluster

df_ks_1 = df_ks.loc[df_ks.clusters == "High Profile"]
df_ks_2 = df_ks.loc[df_ks.clusters == "The Overly Ambitious"]
df_ks_3 = df_ks.loc[df_ks.clusters == "The not-so-interesting"]

lab =list(df_ks_1.groupby("category", as_index=False).count()[["category", "name"]].sort_values(by="name", ascending=False).category)

ma_plot = df_ks_1.groupby("category", as_index=False).count()[["category", "name"]].sort_values(by="name", ascending=False).plot.pie(y="name", autopct="%1.1f%%", figsize=(17,13), labels=lab)
ma_plot.set(ylabel="")
plt.legend("")
plt.title("Category Distribution among High Profile Projects")
plt.show()

# The names of the projects belonging to this cluster makes you realize why the machine made them into their own cluster

df_ks_2[["name", "backers"]].name.head(15)

# Misc EDA

yrs=[]
for i in range(len(df_ks.groupby([df_ks.launched.dt.year, df_ks.launched.dt.month]).count().iloc[1:, :].index)):
    yrs.append(df_ks.groupby([df_ks.launched.dt.year, df_ks.launched.dt.month]).count().iloc[1:, :].index[i][0])

plt.figure(figsize=(12, 8))
sns.lineplot(data=df_ks.groupby([df_ks.launched.dt.year, df_ks.launched.dt.month]).count().iloc[1:, :], x=yrs, y="name")
plt.title("Amount of KS projects launched per year")
plt.xlabel("Year")
plt.ylabel("Amount")
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(data=df_ks.groupby("category", as_index=False).count().sort_values(by="name").head(), x="name", y="category")
plt.title("Least common categories")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(data=df_ks[["name", "usd pledged"]].sort_values(by="usd pledged", ascending=False).head(), x="usd pledged", y="name")
plt.title("Most pledged to project [USD]")
plt.xlabel("USD")
plt.ylabel("Name")
plt.show()