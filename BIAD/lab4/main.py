import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

def generate_data():
    N = 20
    x = np.linspace(0, 1, N)
    noise = np.random.normal(0, 1, N)
    f = lambda x: np.tan(7*x) + (4*x**3 - 2*x + 15) / (18*x**4 + 9*x**2 - 3*x + 1)
    y = f(x) + noise
    return x, y

def split_data(x, y):
    plt.plot(x, y)
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    return x_train, x_test, y_train, y_test

def plot_data(x_train, y_train, x_test, y_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='blue', label='Training Data')
    plt.scatter(x_test, y_test, color='red', label='Test Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Generated Data')
    plt.legend()
    plt.show()
    

def linear_regression_exact(x_train, y_train):
    X = np.vstack([np.ones_like(x_train), x_train]).T
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y_train
    return coefficients

def linear_regression_gd(x_train, y_train, learning_rate=0.01, epochs=1000):
    X = np.vstack([np.ones_like(x_train), x_train]).T
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        gradient = (2/m) * X.T @ (X @ theta - y_train)
        theta -= learning_rate * gradient
    return theta

def plot_regression_lines(x_train, y_train, coefficients_exact, coefficients_gd):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='blue', label='Training Data')
    plt.plot(x_train, coefficients_exact[0] + coefficients_exact[1]*x_train, color='green', label='Exact Solution')
    plt.plot(x_train, coefficients_gd[0] + coefficients_gd[1]*x_train, color='orange', label='Gradient Descent')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Approximations')
    plt.legend()
    plt.show()

def polynomial_regression(x_train, y_train, x_test, y_test):
    degrees = [2, 5, 9]
    mse_train = []
    mse_test = []
    
    for degree in degrees:
        poly_features = PolynomialFeatures(degree)
        X_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
        X_test_poly = poly_features.fit_transform(x_test.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        mse_train.append(mean_squared_error(y_train, y_train_pred))
        mse_test.append(mean_squared_error(y_test, y_test_pred))
    
    return mse_train, mse_test

def plot_mse(mse_train, mse_test):
    degrees = [2, 5, 9]
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, mse_train, label='Training MSE', marker='o')
    plt.plot(degrees, mse_test, label='Test MSE', marker='o')
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('MSE')
    plt.title('MSE vs Degree of Polynomial')
    plt.legend()
    plt.show()

def nonlinear_approximation(x_train, y_train, x_test, y_test):
    def func(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.sin(d * x)
    
    initial_params = [1, 1, 1, 1]
    
    params, _ = curve_fit(func, x_train, y_train, p0=initial_params)
    
    x_smooth = np.linspace(0, 1, 100)
    y_smooth = func(x_smooth, *params)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='blue', label='Training Data')
    plt.plot(x_smooth, y_smooth, color='green', label='Nonlinear Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Improved Nonlinear Approximation')
    plt.legend()
    plt.show()

    y_train_pred = func(x_train, *params)
    y_test_pred = func(x_test, *params)
    
    from sklearn.metrics import mean_squared_error
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    print(f"Nonlinear Function Parameters: {params}")
    print(f"Training MSE: {mse_train:.4f}")
    print(f"Test MSE: {mse_test:.4f}")

def ridge_regression(x_train, y_train, x_test, y_test):
    alphas = np.logspace(-4, 4, 100)
    mse_train = []
    mse_test = []
    
    poly_features = PolynomialFeatures(degree=5)
    X_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
    X_test_poly = poly_features.fit_transform(x_test.reshape(-1, 1))
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_poly, y_train)
        
        y_train_pred = ridge.predict(X_train_poly)
        y_test_pred = ridge.predict(X_test_poly)
        
        mse_train.append(mean_squared_error(y_train, y_train_pred))
        mse_test.append(mean_squared_error(y_test, y_test_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, mse_train, label='Training MSE')
    plt.plot(alphas, mse_test, label='Test MSE')
    plt.xscale('log')
    plt.xlabel('Regularization Strength')
    plt.ylabel('MSE')
    plt.title('Ridge Regression: MSE vs Regularization Strength')
    plt.legend()
    plt.show()


def main():
    x, y = generate_data()
    x_train, x_test, y_train, y_test = split_data(x, y)
    plot_data(x_train, y_train, x_test, y_test)
    
    coefficients_exact = linear_regression_exact(x_train, y_train)
    coefficients_gd = linear_regression_gd(x_train, y_train)
    
    plot_regression_lines(x_train, y_train, coefficients_exact, coefficients_gd)
    
    mse_train, mse_test = polynomial_regression(x_train, y_train, x_test, y_test)
    plot_mse(mse_train, mse_test)
    
    nonlinear_approximation(x_train, y_train, x_test, y_test)
    
    ridge_regression(x_train, y_train, x_test, y_test)
    
if __name__ == "__main__":
    main()
