from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

weather_data = pd.read_csv(r'D:\Python\Projects\Community\ADY Work\Data Science Stuff\Climate Data.csv')
weather_data['Date'] = pd.to_datetime(weather_data['Date'])


x = weather_data.drop(columns=['Date', 'Average temperature (°F)'])
y = weather_data['Average temperature (°F)']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_pred,y_test, label='predict', color='blue')
plt.plot(y_pred,y_pred, label='Actual', color='red')
plt.title('Predictions')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.show()