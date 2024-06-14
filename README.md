# Black-Women-Wage-Gap
Final Project 613

## Introduction
This project aims to emphasize and visualize the wage gap through the lens of black women by using a Linear Regression model.

## Installation
### Prerequisites
Ensure you have the following software is installed:

- Python 3
- pip
### Dependencies
Install the required Python libraries using pip:
```sh
pip install pandas scikit-learn graphviz matplotlib
```
## Usage
### Data Preparation
Load dataset using pandas:
```sh
data = pd.read_csv('black_white_wage_gap.csv')
print(data.head())

X = data.drop(['black_women_median', 'black_women_average'], axis=1)  # Features
y = data['black_women_average']  # Target variable
```

### Train-Test Split
Split the dataset into training and testing sets:
```sh
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training
Linear Regression
```sh
from sklearn.linear_model import LinearRegression
# Train the model
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Contributing
Contributions are welcome!

### Acknowledgements
Thank the contributors of the open-sourse libraries and data that was used in this project.

### Contact
For questions/ support, please contact me at mab857@drexel.edu
