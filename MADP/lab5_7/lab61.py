import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_diabetes, fetch_california_housing, load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

def task1():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix for Iris Dataset')
    plt.show()

def task2():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)

    corr_matrix = df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]

    df_reduced = df.drop(columns=to_drop)

    print(f"Исключенные признаки: {to_drop}")
    print("Оставшиеся признаки после исключения мультиколлинеарных:")
    print(df_reduced.columns)

def task3():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    target = california.target

    corr_with_target = df.apply(lambda x: x.corr(pd.Series(target)))

    n = 5
    top_features = corr_with_target.abs().sort_values(ascending=False).head(n).index

    print(f"{n} признаков с наибольшими абсолютными значениями коэффициента корреляции:")
    print(top_features)

def task4():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    target = diabetes.target

    spearman_corr = df.apply(lambda x: spearmanr(x, target)[0])

    top_spearman_features = spearman_corr.abs().sort_values(ascending=False).head(5).index

    print("Признаки с наибольшими значениями ранговой корреляции Спирмена:")
    print(top_spearman_features)

def task5():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    target = diabetes.target
    print(df.columns)

    # Метод 1: Корреляция Пирсона
    pearson_corr = df.apply(lambda x: x.corr(pd.Series(target)))
    top_pearson_features = pearson_corr.abs().sort_values(ascending=False).head(5).index

    # Метод 2: Корреляция Спирмена
    spearman_corr = df.apply(lambda x: spearmanr(x, target)[0])
    top_spearman_features = spearman_corr.abs().sort_values(ascending=False).head(5).index

    print("Сравнение методов отбора признаков:")
    print(f"Корреляция Пирсона: {top_pearson_features}")
    print(f"Корреляция Спирмена: {top_spearman_features}")

def task6():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    preprocessors = {
        'Original': None,
        'Standardization': StandardScaler(),
        'Normalization': MinMaxScaler(),
        'Log Transformation': FunctionTransformer(np.log1p)
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for ax, (method, preprocessor) in zip(axes.flat, preprocessors.items()):
        if preprocessor:
            df_transformed = preprocessor.fit_transform(df)
        else:
            df_transformed = df
        
        corr_matrix = pd.DataFrame(df_transformed, columns=diabetes.feature_names).corr().abs()

        sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title(f'Correlation Matrix: {method}')

    plt.tight_layout()
    plt.show()


def main():
    print("ЗАДАНИЕ 1")
    #task1()
    print("ЗАДАНИЕ 2")
    #task2()
    print("ЗАДАНИЕ 3")
    #task3()
    print("ЗАДАНИЕ 4")
    #task4()
    print("ЗАДАНИЕ 5")
    task5()
    print("ЗАДАНИЕ 6")
    task6()


if __name__ == "__main__":
    main()