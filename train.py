import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
y = target.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

clf = TabNetRegressor()
clf.fit(
  X_train, y_train,
  eval_set=[(X_valid, y_valid)],
  max_epochs=1000,
  patience=100
)
preds_test = clf.predict(X_test)

print('Test MSE', mean_squared_error(y_test.reshape(-1), preds_test.reshape(-1)))

plt.ylabel('price')
plt.plot(np.arange(y_test.shape[0]), y_test.reshape(-1), label='y_test')
plt.plot(np.arange(y_test.shape[0]), preds_test.reshape(-1), label='preds')
plt.legend()
plt.show()
