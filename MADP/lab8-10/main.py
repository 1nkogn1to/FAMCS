import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt


def task1():
    california = fetch_california_housing()
    X, y = california.data, california.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Task 1 - Mean Squared Error: {mse:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Linear Regression: True Values vs Predictions')
    plt.show()

def task2():
    california = fetch_california_housing()
    X, y = california.data, california.target
    
    model = LinearRegression()
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    
    mean_mse = -np.mean(mse_scores)
    print(f"Task 2 - Mean MSE across all folds: {mse_scores}")

def task3():
    california = fetch_california_housing()
    X, y = california.data, california.target
    
    model = Ridge()
    
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    
    best_alpha = grid_search.best_params_['alpha']
    print(f"Task 3 - Best alpha: {best_alpha}")
    
    best_model = Ridge(alpha=best_alpha)
    best_model.fit(X, y)
    
    y_pred = best_model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    print(f"Task 3 - Mean Squared Error: {mse:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Ridge Regression: True Values vs Predictions')
    plt.show()

def task4():
    california = fetch_california_housing()
    X, y = california.data, california.target
    
    model = Lasso(max_iter=10000)
    
    param_grid = {'alpha': [10, 1, 0.1, 0.01]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    
    best_alpha = grid_search.best_params_['alpha']
    print(f"Task 4 - Best alpha: {best_alpha}")

    best_model = Lasso(alpha=best_alpha, max_iter=10000)
    best_model.fit(X, y)

    y_pred = best_model.predict(X)

    mse = mean_squared_error(y, y_pred)
    print(f"Task 4 - Mean Squared Error: {mse:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Lasso Regression: True Values vs Predictions')
    plt.show()
    
    zero_weights = np.sum(best_model.coef_ == 0)
    print(f"Task 4 - Number of zero weights in Lasso model: {zero_weights}")
    
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X, y)
    ridge_zero_weights = np.sum(ridge_model.coef_ == 0)
    print(f"Task 4 - Number of zero weights in Ridge model: {ridge_zero_weights}")

def task5():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = HuberRegressor()
    
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'epsilon': [1.1, 1.35, 1.5, 1.75, 2.0]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_alpha = grid_search.best_params_['alpha']
    best_epsilon = grid_search.best_params_['epsilon']
    print(f"Task 5 - Best alpha: {best_alpha}, Best epsilon: {best_epsilon}")
    
    best_model = HuberRegressor(alpha=best_alpha, epsilon=best_epsilon, max_iter=10000)
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Task 5 - Mean Squared Error: {mse:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Huber Regressor: True Values vs Predictions')
    plt.show()

def task6():
    california = fetch_california_housing()
    X, y = california.data, california.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ridge = Ridge()
    lasso = Lasso(max_iter=10000)
    
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    
    ridge_coef_orig = ridge.coef_
    lasso_coef_orig = lasso.coef_
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    
    ridge_coef_scaled = ridge.coef_
    lasso_coef_scaled = lasso.coef_
    
    ridge_pred_scaled = ridge.predict(X_test_scaled)
    lasso_pred_scaled = lasso.predict(X_test_scaled)
    ridge_mse_scaled = mean_squared_error(y_test, ridge_pred_scaled)
    lasso_mse_scaled = mean_squared_error(y_test, lasso_pred_scaled)
    
    normalizer = MinMaxScaler()
    X_train_normalized = normalizer.fit_transform(X_train)
    X_test_normalized = normalizer.transform(X_test)
    
    ridge.fit(X_train_normalized, y_train)
    lasso.fit(X_train_normalized, y_train)
    
    ridge_coef_normalized = ridge.coef_
    lasso_coef_normalized = lasso.coef_
    
    ridge_pred_normalized = ridge.predict(X_test_normalized)
    lasso_pred_normalized = lasso.predict(X_test_normalized)
    ridge_mse_normalized = mean_squared_error(y_test, ridge_pred_normalized)
    lasso_mse_normalized = mean_squared_error(y_test, lasso_pred_normalized)
    
    print("Task 6 - Ridge Coefficients Comparison")
    print(f"Original: {ridge_coef_orig}")
    print(f"Standardized: {ridge_coef_scaled}, MSE: {ridge_mse_scaled:.4f}")
    print(f"Normalized: {ridge_coef_normalized}, MSE: {ridge_mse_normalized:.4f}\n")
    
    print("Task 6 - Lasso Coefficients Comparison")
    print(f"Original: {lasso_coef_orig}")
    print(f"Standardized: {lasso_coef_scaled}, MSE: {lasso_mse_scaled:.4f}")
    print(f"Normalized: {lasso_coef_normalized}, MSE: {lasso_mse_normalized:.4f}")

def task7():
    X, y = make_regression(n_samples=100, n_features=1000, noise=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    alphas = [0.01, 0.1, 1, 10, 100]
    
    ridge_results = []
    lasso_results = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge.predict(X_test_scaled)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        ridge_results.append((alpha, mse_ridge, ridge.coef_))
        
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict(X_test_scaled)
        mse_lasso = mean_squared_error(y_test, y_pred_lasso)
        lasso_results.append((alpha, mse_lasso, lasso.coef_))
    
    for alpha, mse, coefs in ridge_results:
        non_zero_coefs = np.sum(coefs != 0)
        print(f"Ridge Regression - Alpha: {alpha}, MSE: {mse:.4f}, Non-zero coefficients: {non_zero_coefs}")
    
    print("\n")
    
    for alpha, mse, coefs in lasso_results:
        non_zero_coefs = np.sum(coefs != 0)
        print(f"Lasso Regression - Alpha: {alpha}, MSE: {mse:.4f}, Non-zero coefficients: {non_zero_coefs}")
    
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    for alpha, _, coefs in ridge_results:
        plt.plot(coefs, label=f'alpha={alpha}')
    plt.title('Ridge Regression Coefficients')
    plt.xlabel('Features')
    plt.ylabel('Coefficient value')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for alpha, _, coefs in lasso_results:
        plt.plot(coefs, label=f'alpha={alpha}')
    plt.title('Lasso Regression Coefficients')
    plt.xlabel('Features')
    plt.ylabel('Coefficient value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def task8():
    X, y = make_regression(n_samples=100, n_features=1000, noise=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    alphas = [0.01, 0.1, 1, 10, 100]
    
    enet_results = []
    
    for l1_ratio in l1_ratios:
        for alpha in alphas:
            enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            enet.fit(X_train_scaled, y_train)
            y_pred = enet.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            enet_results.append((alpha, l1_ratio, mse, enet.coef_))
    
    for alpha, l1_ratio, mse, coefs in enet_results:
        non_zero_coefs = np.sum(coefs != 0)
        print(f"ElasticNet - Alpha: {alpha}, L1 ratio: {l1_ratio}, MSE: {mse:.4f}, Non-zero coefficients: {non_zero_coefs}")
    
    plt.figure(figsize=(14, 7))
    
    for l1_ratio in l1_ratios:
        plt.plot([], [], label=f'L1 ratio={l1_ratio}')
        for alpha, _, _, coefs in enet_results:
            if alpha in alphas:
                plt.plot(coefs, label=f'alpha={alpha}', alpha=0.3)
    
    plt.title('ElasticNet Regression Coefficients')
    plt.xlabel('Features')
    plt.ylabel('Coefficient value')
    plt.legend()
    plt.show()

def task9():
    california = fetch_california_housing()
    X, y = california.data, california.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lasso_selector = SelectFromModel(Lasso(alpha=0.1, max_iter=10000))
    lasso_selector.fit(X_train_scaled, y_train)
    X_train_selected = lasso_selector.transform(X_train_scaled)
    X_test_selected = lasso_selector.transform(X_test_scaled)
    
    ridge_all = Ridge(alpha=1.0)
    ridge_all.fit(X_train_scaled, y_train)
    y_pred_ridge_all = ridge_all.predict(X_test_scaled)
    mse_ridge_all = mean_squared_error(y_test, y_pred_ridge_all)
    
    lasso_all = Lasso(alpha=0.1, max_iter=10000)
    lasso_all.fit(X_train_scaled, y_train)
    y_pred_lasso_all = lasso_all.predict(X_test_scaled)
    mse_lasso_all = mean_squared_error(y_test, y_pred_lasso_all)
    
    ridge_selected = Ridge(alpha=1.0)
    ridge_selected.fit(X_train_selected, y_train)
    y_pred_ridge_selected = ridge_selected.predict(X_test_selected)
    mse_ridge_selected = mean_squared_error(y_test, y_pred_ridge_selected)
    
    lasso_selected = Lasso(alpha=0.1, max_iter=10000)
    lasso_selected.fit(X_train_selected, y_train)
    y_pred_lasso_selected = lasso_selected.predict(X_test_selected)
    mse_lasso_selected = mean_squared_error(y_test, y_pred_lasso_selected)
    
    print("Task 9 - Mean Squared Error Comparison")
    print(f"Ridge Regression on All Features: MSE = {mse_ridge_all:.4f}")
    print(f"Lasso Regression on All Features: MSE = {mse_lasso_all:.8f}")
    print(f"Ridge Regression on Selected Features: MSE = {mse_ridge_selected:.4f}")
    print(f"Lasso Regression on Selected Features: MSE = {mse_lasso_selected:.8f}")
    
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)
    plt.plot(ridge_all.coef_, label='Ridge All Features')
    plt.plot(ridge_selected.coef_, label='Ridge Selected Features', linestyle='--')
    plt.title('Ridge Regression Coefficients')
    plt.xlabel('Features')
    plt.ylabel('Coefficient value')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(lasso_all.coef_, label='Lasso All Features')
    plt.plot(lasso_selected.coef_, label='Lasso Selected Features', linestyle='--')
    plt.title('Lasso Regression Coefficients')
    plt.xlabel('Features')
    plt.ylabel('Coefficient value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def task10():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    print(f"Linear Regression - Mean Squared Error: {mse_lr:.4f}")
    
    huber_model = HuberRegressor(max_iter=1000)
    huber_model.fit(X_train, y_train)
    y_pred_huber = huber_model.predict(X_test)
    mse_huber = mean_squared_error(y_test, y_pred_huber)
    print(f"Huber Regressor - Mean Squared Error: {mse_huber:.4f}")
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_lr, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Linear Regression: True Values vs Predictions')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_huber, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Huber Regressor: True Values vs Predictions')
    
    plt.tight_layout()
    plt.show()

def main():
    #task1()
    #task2()
    #task3()
    #task4()
    #task5()
    #task6()
    #task7()
    #task8()
    task9()
    task10()
    
if __name__ == "__main__":
    main()
