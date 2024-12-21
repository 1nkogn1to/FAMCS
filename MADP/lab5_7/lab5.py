import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.datasets as dat
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
import statsmodels.api as sm

def task1():
    wine_data = dat.load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    
    corr_matrix = df.corr()
    
    plt.figure(figsize=(12, 8))
    sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix for Wine Dataset')
    plt.show()

def task2():
    wine_data = dat.load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    target = wine_data.target

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_

    feature_importances_df = pd.DataFrame({
        'feature': df.columns,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sb.barplot(x='importance', y='feature', data=feature_importances_df)
    plt.title('Feature Importances for Wine Dataset')
    plt.show()

def task3():
    california = dat.fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    target = california.target

    df['MedHouseVal'] = target

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    selected_features = model.pvalues[model.pvalues < 0.05].index
    selected_features = selected_features.drop('const')

    print("Выбранные признаки на основе p-value:")
    print(selected_features)

def task4():
    digits = dat.load_digits()
    df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
    target = digits.target

    mutual_info = mutual_info_classif(df, target)

    mutual_info_df = pd.DataFrame({
        'feature': df.columns,
        'mutual_info': mutual_info
    }).sort_values(by='mutual_info', ascending=False)

    plt.figure(figsize=(12, 8))
    sb.barplot(x='mutual_info', y='feature', data=mutual_info_df)
    plt.title('Важность признаков на основе взаимной информации:')
    plt.show()

def task5():
    digits = dat.load_digits()
    df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
    target = digits.target

    model = SVC(kernel="linear")

    rfe = RFE(estimator=model, n_features_to_select=10, step=1)
    fit = rfe.fit(df, target)

    rfe_df = pd.DataFrame({
        'feature': df.columns,
        'ranking': fit.ranking_
    }).sort_values(by='ranking', ascending=True)

    plt.figure(figsize=(12, 8))
    sb.barplot(x='ranking', y='feature', data=rfe_df)
    plt.title('Ранжирование признаков на основе RFE:')
    plt.show()

def task6():
    breast_cancer = dat.load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    target = breast_cancer.target
    print(df.columns)

    model = LogisticRegression(max_iter=10000, random_state=42)

    # Метод 1: Взаимная информация
    mutual_info = mutual_info_classif(df, target)
    selected_features_mi = df.columns[mutual_info > 0.1]
    X_mi = df[selected_features_mi]

    # Метод 2: p-value
    X_pval = sm.add_constant(df)
    model_ols = sm.OLS(target, X_pval).fit()
    selected_features_pval = model_ols.pvalues[model_ols.pvalues < 0.05].index.drop('const')
    X_pval = df[selected_features_pval]

    # Метод 3: RFE
    rfe = RFE(estimator=model, n_features_to_select=10, step=1)
    fit = rfe.fit(df, target)
    selected_features_rfe = df.columns[fit.support_]
    X_rfe = df[selected_features_rfe]

    # Метод 4: Корреляционные матрицы
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    selected_features_corr = [column for column in upper.columns if any(upper[column] > 0.6)]
    X_corr = df[selected_features_corr]

    # Метод 5: Важность признаков в случайном лесе
    rf = RandomForestClassifier(random_state=42)
    rf.fit(df, target)
    importances = rf.feature_importances_
    selected_features_rf = df.columns[importances > np.mean(importances)]
    X_rf = df[selected_features_rf]

    scores_mi = cross_val_score(model, X_mi, target, cv=5)
    scores_pval = cross_val_score(model, X_pval, target, cv=5)
    scores_rfe = cross_val_score(model, X_rfe, target, cv=5)
    scores_corr = cross_val_score(model, X_corr, target, cv=5)
    scores_rf = cross_val_score(model, X_rf, target, cv=5)

    print("Производительность моделей на различных подмножествах признаков:")
    print(f"Взаимная информация: Среднее значение точности = {scores_mi.mean():.4f}")
    print(f"p-value: Среднее значение точности = {scores_pval.mean():.4f}")
    print(f"RFE: Среднее значение точности = {scores_rfe.mean():.4f}")
    print(f"Корреляционные матрицы: Среднее значение точности = {scores_corr.mean():.4f}")
    print(f"Важность признаков в случайном лесе: Среднее значение точности = {scores_rf.mean():.4f}")


def main():
    #task1()
    #task2()
    #task3()
    #task4()
    task5()
    task6()

if __name__ == "__main__":
    main()
