from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


stock_data = pd.read_csv(r"D:\Python\Projects\Community\ADY Work\Data Science Stuff\All Stocks.csv")
stock_data.head()
stock_data.isnull().sum()
stock_data.describe()
stock_data = stock_data.dropna()
stock_data['Date'] = pd.to_datetime(stock_data['Date'])


X = stock_data[['High', 'Low']]
y = stock_data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)
print(mean_squared_error(y_test, predictions))

plt.figure(figsize=(10, 5))
plt.scatter(y_test, predictions, color='blue', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Actual')
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.title('Linear Regression: Actual vs Predicted Close Prices')
plt.legend()
plt.show()