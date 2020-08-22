import pandas as pd
import numpy as np
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
from sklearn.linear_model import LinearRegression

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


