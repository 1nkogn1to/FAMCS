import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


def task1(df):
    z_scores = np.abs(zscore(df.drop(columns=['Class', 'Time', 'Amount'])))
    threshold = 3
    anomalies_zscore = (z_scores > threshold)
    anomalies_count_zscore = anomalies_zscore.sum(axis=0) / len(anomalies_zscore['V1']) * 100

    print("Процент аномалий по каждому столбцу (метод Z-score):")
    print(anomalies_count_zscore)


def task2(df):
    df_cleaned = df.drop(columns=['Class', 'Time', 'Amount'])
    Q1 = df_cleaned.quantile(0.25)
    Q3 = df_cleaned.quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5
    anomalies_iqr = ((df_cleaned < (Q1 - threshold * IQR)) | (df_cleaned > (Q3 + threshold * IQR)))
    anomalies_count_iqr = anomalies_iqr.sum(axis=0) / len(anomalies_iqr['V1']) * 100

    print("Процент аномалий по каждому столбцу (метод IQR):")
    print(anomalies_count_iqr)

def count_anomalies(group):
    anomalies_count = {}
    for column in ['MaxTemp', 'MinTemp', 'MeanTemp']:
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = group[(group[column] < lower_bound) | (group[column] > upper_bound)]
        anomalies_count[column] = anomalies.shape[0]
    return pd.Series(anomalies_count)


def task3(df):
    columns_needed = ['STA', 'MaxTemp', 'MinTemp', 'MeanTemp']
    df_cleaned = df[columns_needed]

    grouped = df_cleaned.groupby('STA')

    anomalies_counts = grouped.apply(count_anomalies).reset_index()

    print("Количество аномалий по каждому столбцу для каждой группы (метод IQR):")
    print(anomalies_counts)


def task4(df):
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)

    imputer = KNNImputer(n_neighbors=5)

    data_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    X = data_imputed.drop('Outcome', axis=1)
    y = data_imputed['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

def plot_survival(data, title):
    plt.figure(figsize=(8, 5))
    sb.countplot(data=data, x='survived', hue='survived', palette='pastel')
    plt.title(title)
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()

def task5():
    titanic = sb.load_dataset('titanic')

    titanic_dropped = titanic.dropna()

    age_mean = titanic['age'].mean()  
    titanic_mean_filled = titanic.copy()
    titanic_mean_filled['age'] = titanic_mean_filled['age'].fillna(age_mean)

    age_median = titanic['age'].median()
    titanic_median_filled = titanic.copy()
    titanic_median_filled['age'] = titanic_median_filled['age'].fillna(age_median)

    titanic_mode_filled = titanic.copy()
    titanic_mode_filled['embarked'] = titanic_mode_filled['embarked'].fillna(titanic['embarked'].mode()[0])

    plot_survival(titanic, 'До обработки')
    plot_survival(titanic_dropped, 'После удаления строк')
    plot_survival(titanic_mean_filled, 'После заполнения средним')
    plot_survival(titanic_median_filled, 'После заполнения медианой')
    plot_survival(titanic_mode_filled, 'После заполнения модой')

def task6(df):
    print("Оригинальные данные:")
    print(df.head())

    df_min_max = df.copy()
    df_standard = df.copy()
    df_normalized = df.copy()

    # Применение Мин-Макс нормализации
    min_max_scaler = MinMaxScaler()
    df_min_max[df.columns] = min_max_scaler.fit_transform(df)

    # Применение стандартизации (Z-score)
    standard_scaler = StandardScaler()
    df_standard[df.columns] = standard_scaler.fit_transform(df)

    # Применение нормализации (L2-нормализация)
    normalizer = Normalizer()
    df_normalized[df.columns] = normalizer.fit_transform(df)

    column = 'alcohol'
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.hist(df[column], bins=30, alpha=0.75, color='blue')
    plt.title('Исходные данные')
    plt.xlabel(column)
    plt.ylabel('Частота')
    
    plt.subplot(2, 2, 2)
    plt.hist(df_min_max[column], bins=30, alpha=0.75, color='green')
    plt.title('Мин-Макс нормализация')
    plt.xlabel(column)
    plt.ylabel('Частота')
    
    plt.subplot(2, 2, 3)
    plt.hist(df_standard[column], bins=30, alpha=0.75, color='red')
    plt.title('Стандартизация (Z-score)')
    plt.xlabel(column)
    plt.ylabel('Частота')

    plt.subplot(2, 2, 4)
    plt.hist(df_normalized[column], bins=30, alpha=0.75, color='purple')
    plt.title('Нормализация (L2-нормализация)')
    plt.xlabel(column)
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.show()

def task7(df):
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

    correlation_matrix = df[numerical_features].corr()

    plt.figure(figsize=(20, 10))
    sb.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.2)
    plt.title('Корреляционная матрица')
    plt.show()

    threshold = 0.4
    strong_correlations = correlation_matrix[correlation_matrix.abs() > threshold]
    print("Сильные корреляции:")
    print(strong_correlations)

def task8(df):
    print("Исходные данные:")
    print(df[['tweet_id', 'text']].head())

    # Загрузка необходимых ресурсов NLTK
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Определение стоп-слов, стеммера и лемматизатора
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.lower()
        words = [word for word in text.split() if word not in stop_words]
        words = [stemmer.stem(word) for word in words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    df['processed_text'] = df['text'].apply(preprocess_text)

    print("\nДанные после предобработки:")
    print(df[['tweet_id', 'processed_text']].head())

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])

    print("\nПример векторизации (первая строка):")
    #print(vectorizer.get_feature_names_out())
    test_names = vectorizer.get_feature_names_out()
    test_nums = X.toarray()[4]
    size = len(test_nums)
    count = 0
    for i in range(size):
        if test_nums[i] == 0:
            test_nums[i], test_nums[-1] = test_nums[-1], test_nums[i]
            test_names[i], test_names[-1] = test_names[-1], test_names[i]
            size -= 1
        else:
            count += 1

    print(test_names[:10])
    print(test_nums[:10])
    print(count)

    #print(X.toarray()[0][:40])

def main():
    df_task1_2 = pd.read_csv('info/creditcard.csv')
    #task1(df_task1_2)
    #task2(df_task1_2)

    df_task3 = pd.read_csv('info/Summary of Weather.csv', low_memory=False)
    #task3(df_task3)

    df_task4 = pd.read_csv('info/diabetes.csv')
    #task4(df_task4)

    
    #task5()

    df_task6 = pd.read_csv('info/winequality-red.csv')
    #task6(df_task6)

    df_task7 = pd.read_csv('info/train.csv')
    #task7(df_task7)

    df_task8 = pd.read_csv('info/Tweets.csv')
    task8(df_task8)

if __name__ == "__main__":
    main()