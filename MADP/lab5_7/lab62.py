import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing, load_wine
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def task1():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = iris.target

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)

    rfe = RFE(estimator=model, n_features_to_select=2)
    rfe.fit(X_train, y_train)
    
    selected_features = df.columns[rfe.support_]
    print(f"Выбранные признаки: {selected_features}")

    X_train_rfe = X_train[selected_features]
    X_test_rfe = X_test[selected_features]
    model.fit(X_train_rfe, y_train)
    score_rfe = model.score(X_test_rfe, y_test)
    print(f"Производительность модели с выбранными признаками: {score_rfe:.4f}")

    model.fit(X_train, y_train)
    score_all = model.score(X_test, y_test)
    print(f"Производительность модели со всеми признаками: {score_all:.4f}")

def task2():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    target = california.target

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)

    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=3, direction='forward', n_jobs=-1)
    sfs.fit(X_train, y_train)
    
    selected_features = df.columns[sfs.get_support()]
    print(f"Выбранные признаки: {selected_features}")

    model.fit(X_train[selected_features], y_train)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sb.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importances for California Housing Dataset')
    plt.show()

def task3():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = wine.target

    model = SVC(kernel='linear')

    rfe = RFE(estimator=model, n_features_to_select=5)
    
    scores = cross_val_score(rfe, df, target, cv=5)
    print(f"Средняя точность при кросс-валидации: {scores.mean():.4f}")

def task4():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = iris.target

    model = LogisticRegression(max_iter=50)

    # Метод 1: RFE
    rfe = RFE(estimator=model, n_features_to_select=2)
    scores_rfe = cross_val_score(rfe, df, target, cv=5)
    print(f"Средняя точность при использовании RFE: {scores_rfe.mean():.4f}")

    # Метод 2: Sequential Feature Selector
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=2, direction='forward')
    sfs.fit(df, target)
    selected_features = df.columns[sfs.get_support()]
    df_selected = df[selected_features]
    
    scores_sfs = cross_val_score(model, df_selected, target, cv=5)
    print(f"Средняя точность при использовании Sequential Feature Selector: {scores_sfs.mean():.4f}")

def task5():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = wine.target

    model = RandomForestClassifier(random_state=42)

    # Метод-обертка для отбора признаков
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=5, direction='forward')
    sfs.fit(df, target)
    
    selected_features = df.columns[sfs.get_support()]
    print(f"Выбранные признаки: {selected_features}")

    scores_selected = cross_val_score(model, df[selected_features], target, cv=5)
    print(f"Средняя точность с выбранными признаками: {scores_selected.mean():.4f}")

    scores_all = cross_val_score(model, df, target, cv=5)
    print(f"Средняя точность со всеми признаками: {scores_all.mean():.4f}")

def main():
    print("ЗАДАНИЕ 1")
    #task1()
    print("ЗАДАНИЕ 2")
    #task2()
    print("ЗАДАНИЕ 3")
    #task3()
    print("ЗАДАНИЕ 4")
    task4()
    print("ЗАДАНИЕ 5")
    #task5()

if __name__ == "__main__":
    main()
