import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import ttest_ind


def task2(dataframe):
    group_counts = dataframe['group'].value_counts()
    print("Количество студентов в каждой группе:\n", group_counts)

    sex_counts = dataframe['sex'].value_counts()
    print("Количество студентов по полу:\n", sex_counts)

    work_status_counts = dataframe['work_status'].value_counts(normalize=True) * 100
    print("Доля работающих студентов:\n", work_status_counts)

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    sb.countplot(x='sex', data=dataframe)
    plt.title("Распределение по полу")

    plt.subplot(2, 1, 2)
    sb.countplot(x='work_status', data=dataframe)
    plt.title("Распределение по статусу работы")

    plt.tight_layout()
    plt.show()

def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def bk(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def binomial_test(successes_1, trials_1, successes_2, trials_2):
    total_successes = successes_1 + successes_2
    total_trials = trials_1 + trials_2

    p_pooled = total_successes / total_trials

    p1 = successes_1 / trials_1

    p_value = 0.0

    for k in range(min(successes_1, successes_2) + 1):
        prob = bk(trials_1, k) * (p_pooled ** k) * ((1 - p_pooled) ** (trials_1 - k))
        p_value += prob

    return 2 * min(p_value, 1 - p_value)


def task3(dataframe):
    
    plt.figure(figsize=(10, 6))
    sb.histplot(dataframe['visit_freq'], bins=10, kde=True)
    plt.show()

    porog = 5

    working_students = dataframe[dataframe['work_status'] != 'Нет']
    non_working_students = dataframe[dataframe['work_status'] == 'Нет']

    successes_working = (working_students['visit_freq'] > porog).sum()
    trials_working = len(working_students)

    successes_non_working = (non_working_students['visit_freq'] > porog).sum()
    trials_non_working = len(non_working_students)

    p_value = binomial_test(successes_working, trials_working, successes_non_working, trials_non_working)
    print("p-value для гипотезы о влиянии статуса работы на посещаемость:", p_value)

    male_students = dataframe[dataframe['sex'] == 'М']
    female_students = dataframe[dataframe['sex'] == 'Ж']

    successes_male = (male_students['visit_freq'] > porog).sum()
    trial_male = len(male_students)

    successes_female = (female_students['visit_freq'] > porog).sum()
    trial_female = len(female_students)

    p_value = binomial_test(successes_male, trial_male, successes_female, trial_female)
    print("p-value для гипотезы о влиянии пола на посещаемость:", p_value)

def task4(dataframe):

    working_students = dataframe[dataframe['work_status'] != 'Нет']
    non_working_students = dataframe[dataframe['work_status'] == 'Нет']

    # Гипотеза: Рабочий статус влияет на случайное число
    usd_pred_working = working_students['random_value'].dropna()
    usd_pred_non_working = non_working_students['random_value'].dropna()

    t_stat, p_value = ttest_ind(usd_pred_working, usd_pred_non_working)
    print("p-value для гипотезы о влиянии статуса работы на прогноз рандомного числа:", p_value)

    # Гипотеза: Пол влияет на случайное число
    males = dataframe[dataframe['group'] == '12']['random_value'].dropna()
    females = dataframe[dataframe['group'] != '12']['random_value'].dropna()

    t_stat, p_value = ttest_ind(males, females)
    print("p-value для гипотезы о влиянии пола обучения на прогноз рандомного числа:", p_value)

def main():

    # task 1
    df = pd.read_csv("info/data.csv")
    df = df[df['is_stud'] == 'Да']
    print(df.head())

    task2(df)
    task3(df)
    task4(df)

if __name__ == "__main__":
    main()