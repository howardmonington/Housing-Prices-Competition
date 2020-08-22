import pandas as pd
import numpy as np
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\Housing Prices\train.csv')
test_df = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\Housing Prices\test.csv')
sample_submission = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\Housing Prices\sample_submission.csv')

# training dataframe has 1,460 rows, 81 columns
print(f"The training dataframe has {train_df.shape[0]} rows and {train_df.shape[1]} columns")
print(f"The test dataframe has {test_df.shape[0]} rows and {test_df.shape[1]} columns")



# Just going to pick a few features to use with no null values: 
# LotArea, YearBuilt, BedroomAbvGr, KitchenAbvGr, Fireplaces, SalePrice (to predict)

cols_to_use = ['LotArea','YearBuilt','BedroomAbvGr','KitchenAbvGr','Fireplaces']

train = train_df[cols_to_use]
test = test_df[cols_to_use]
y = train_df['SalePrice']

train.boxplot(column = 'LotArea', fontsize = 10, grid = True)
train.boxplot(column = 'YearBuilt')
train.boxplot(column = 'BedroomAbvGr')

# From this, we can see the mean of all of the columns. LotArea and YearBuilt will need to be scaled down
train.describe(percentiles = [.25, .5, .75, .9])

cols_to_scale = train.iloc[:,:2]
cols_to_scale.head()

sc = StandardScaler()
cols_to_scale2 = sc.fit_transform(cols_to_scale)
#cols_to_scale2.describe()
cols_to_scale2 = pd.DataFrame(cols_to_scale2)
cols_to_scale2.head()
cols_to_scale2 = cols_to_scale2.rename({0:"LotArea",1:"YearBuilt"}, axis = 1)
cols_to_scale2.head()

train = train.drop(['LotArea','YearBuilt'], axis = 1)

cols_to_concat = [cols_to_scale2, train]
train = pd.concat(cols_to_concat, axis = 1)
train.head()

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
predictions = pd.Series(predictions)
mean_squared_error(y_test, predictions)

