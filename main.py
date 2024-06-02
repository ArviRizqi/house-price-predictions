import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Membuat dataframe dari file csv
df = pd.read_csv('kc_house_data.csv')

# Memfilter kolom yang akan digunakan
use_df = df.drop(["id", "date", "waterfront", "view", "sqft_above", "sqft_basement", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"], axis=1)

# Cek Missing Values dan duplicate
use_df.isna().sum()
use_df.duplicated().sum()

# Cek info dan type data
use_df.info()

# Cek deskripsi dataframe
use_df.describe()

# Melihat bentuk dataframe
use_df.shape

# Mulai membersihkan data null dan duplicate
# Cek data 'bedrooms'
(use_df["bedrooms"] > 10).sum()
# Ada 2 data yang muncul, karena kamar lebih dari 10 kurang masuk akal maka kita ganti dengan asumsi salah ketik.
print("Total 11 kamar terdapat sejumlah:", (use_df["bedrooms"] == 11).sum())
print("Total 33 kamar terdapat sejumlah:", (use_df["bedrooms"] == 33).sum())
# Ada 1 data 11 dan 1 data 33

# Cek Distribusi Data bedrooms
plt.figure(figsize=(7,7))
ax = sns.countplot(data=use_df, x='bedrooms')
for i in ax.containers:
  ax.bar_label(i,)
plt.title('Distribution of Bed', fontsize=16)
plt.show();

# Cek Distribusi Data bathrooms
plt.figure(figsize=(10, 10))
ax = sns.countplot(data=use_df, x='bathrooms')
for i in ax.containers:
  ax.bar_label(i,)
plt.title('Distribution of Bathrooms', fontsize=16)
plt.show();

# Cek Korelasi
correlation = use_df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, cmap="flare", annot=True, fmt='.2f');

# Lihat Korelasi dengan Sort Values
use_df.corr()["price"].sort_values()

# Distribusi Data Berdasarkan Harga
sns.histplot(data=df, x="price")
plt.title('Histogram of Price')
plt.show();

# Visualisasi scatter plot sqft_living vs price
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sqft_living', y='price', data=use_df)
plt.title('Scatter Plot of Sqft_living vs Price')
plt.xlabel('Sqft_living')
plt.ylabel('Price')
plt.show()

# Visualisasi box plot grade vs price
plt.figure(figsize=(8, 6))
sns.boxplot(x='grade', y='price', data=use_df)
plt.title('Box Plot of Grade vs Price')
plt.xlabel('Grade')
plt.ylabel('Price')
plt.show()

# Cek Data Outlier
column = ["bedrooms", "bathrooms", "sqft_living", "condition", "grade", "price"]
sns.boxplot(use_df[column])
plt.xticks([1, 2, 3, 4, 5, 6], column)
plt.title('Cek Data Outlier')
plt.show();

# Handling Missing Values
use_df.dropna(how='any', inplace = True)
use_df.isnull().sum()

# Handling Duplicated Values
use_df.drop_duplicates(inplace=True)

# Replace nilai 11 & 33 pada column bedrooms
use_df["bedrooms"] = use_df["bedrooms"].replace(11,1)
use_df["bedrooms"] = use_df["bedrooms"].replace(33,3)

# Cek lagi untuk memastikan
print("Total 11 kamar terdapat sejumlah:", (use_df["bedrooms"] == 11).sum())
print("Total 33 kamar terdapat sejumlah:", (use_df["bedrooms"] == 33).sum())

# Convert tipe data "bathrooms" float to int
use_df["bathrooms"].astype('int64')

# Cek data outliuer dari harga
sns.boxplot(use_df["price"])
plt.show()

# Cek Distribusi Dari Harga
sns.displot(use_df["price"]);

# Handling Outlier
Q1 = use_df["price"].quantile(0.25)
Q3 = use_df["price"].quantile(0.75)
IQR = Q3 - Q1

Q1, Q3, IQR

upper = Q3 + (1.5 * IQR)
lower = Q1 - (1.5 * IQR)
lower, upper

use_df.loc[(use_df['price'] > upper) | (use_df['price'] < lower)]

use_df2 = use_df.loc[(use_df['price'] < upper) & (use_df['price'] > lower)]
print('Dataframe sebelum menghapus outliers (df lama): ', len(use_df))
print('Dataframe setelah menghapus outliers: ', len(use_df2))
print('Total outliers: ', len(use_df) - len(use_df2))

# Cek Outlier price setelah handling outlier
sns.boxplot(use_df2['price']);

# import LinearRegression dan train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cek dataframe setelah di handling
use_df2.head()

# Mulai membuat data train
X = use_df2.drop("price", axis=1)
Y = use_df2["price"]

X.head()
Y.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

y_preds = lin_reg.predict(X_test)

lin_reg.score(X_test, Y_test)

from sklearn.metrics import r2_score
r2_score(Y_test, y_preds)

from sklearn.model_selection import cross_val_score
crossval_r2 = cross_val_score(lin_reg, X, Y, cv=5, scoring=None)
crossval_r2

crossval_r2.mean()
price_predict = lin_reg.predict([[4, 2, 2000, 3000, 2, 4, 8, 2007, 0]])

print('''
Prediksi harga dengan kriteria sebagai berikut:

bedrooms = 4
bathrooms = 2
sqft_living = 2000 sqft
sqft_lot = 3000 sqft
floors = 2
condition = 4
grade = 8
yr_built = 2007
yr_renovated = -

adalah:
''')
print(price_predict)

