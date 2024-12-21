import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier


def task1():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = iris.target

    # Применение PCA для снижения размерности до 2 компонент
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)

    plt.figure(figsize=(10, 6))
    plt.scatter(components[:, 0], components[:, 1], c=target, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA on Iris Dataset')
    plt.colorbar()
    plt.show()

def task2():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = wine.target

    # Применение PCA для снижения размерности до 2 компонентов
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df)

    # Применение Factor Analysis для снижения размерности до 2 факторов
    fa = FactorAnalysis(n_components=2)
    fa_components = fa.fit_transform(df)

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=target, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA on Wine Dataset')
    plt.colorbar()
    plt.show()

    # Визуализация результатов Factor Analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(fa_components[:, 0], fa_components[:, 1], c=target, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.title('Factor Analysis on Wine Dataset')
    plt.colorbar()
    plt.show()

def task3():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    target = california.target

    preprocessors = {
        'Original': None,
        'Standardization': StandardScaler(),
        'Normalization': MinMaxScaler()
    }

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    for ax, (method, preprocessor) in zip(axes, preprocessors.items()):
        if preprocessor:
            df_transformed = preprocessor.fit_transform(df)
        else:
            df_transformed = df

        pca = PCA(n_components=2)
        components = pca.fit_transform(df_transformed)

        ax.scatter(components[:, 0], components[:, 1], c=target, cmap='viridis', edgecolor='k', s=50)
        ax.set_title(f'PCA with {method}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')

    plt.tight_layout()
    plt.show()

def task4():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = iris.target

    # Применение Linear Discriminant Analysis (LDA)
    lda = LDA(n_components=2)
    lda_components = lda.fit_transform(df, target)

    # Применение Quadratic Discriminant Analysis (QDA)
    # QDA не используется для снижения размерности, поэтому просто обучим и визуализируем предсказания
    qda = QDA()
    qda.fit(df, target)
    qda_predictions = qda.predict(df)

    plt.figure(figsize=(10, 6))
    plt.scatter(lda_components[:, 0], lda_components[:, 1], c=target, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.title('LDA on Iris Dataset')
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=qda_predictions, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('QDA Predictions on Iris Dataset')
    plt.colorbar()
    plt.show()

def task5():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = wine.target

    # Применение PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df)
    df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])

    # Применение LDA
    lda = LDA(n_components=2)
    lda_components = lda.fit_transform(df, target)
    df_lda = pd.DataFrame(lda_components, columns=['LDA1', 'LDA2'])

    model = RandomForestClassifier(random_state=42)

    # Оценка модели с PCA
    scores_pca = cross_val_score(model, df_pca, target, cv=5)
    print(f"Средняя точность модели с PCA: {scores_pca.mean():.4f}")

    # Оценка модели с LDA
    scores_lda = cross_val_score(model, df_lda, target, cv=5)
    print(f"Средняя точность модели с LDA: {scores_lda.mean():.4f}")

def main():
    task1()
    task2()
    task3()
    task4()
    task5()

if __name__ == "__main__":
    main()