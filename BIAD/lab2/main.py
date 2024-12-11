from scipy.stats import norm, expon, gamma, chi2, chi2_contingency
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def remove_superextramegalargedata(dataframe, column):
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    diff = q3 - q1
    ub = q3 + 1.5 * diff
    return dataframe[dataframe[column] < ub]

def chi_square_test(observed, expected):
    return np.sum((observed - expected) ** 2 / expected)

def check_hypothesis(chi2_stat, critical_value):
    return chi2_stat < critical_value

def main():

    # Задание 1
    df = pd.read_csv("info/train.csv")
    print(len(df["GrLivArea"]))
    df = remove_superextramegalargedata(df, "GrLivArea")
    print(len(df["GrLivArea"]))

    # Задание 2
    data = df["GrLivArea"]

    mu, std = norm.fit(data)
    le, se = expon.fit(data)
    a, lg, sg = gamma.fit(data)

    print(f"Нормальное распределение: mu = {mu}, std = {std}")
    print(f"Экспоненциальное распределение: loc = {le}, scale = {se}")
    print(f"Гамма-распределение: a = {a}, loc = {lg}, scale = {sg}")

    # Задание 3
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, label='Данные')

    x = np.array(sorted(data))
    plt.plot(x, norm.pdf(x, mu, std), label='Нормальное распределение')
    plt.plot(x, expon.pdf(x, le, se), label='Экспоненциальное распределение')
    plt.plot(x, gamma.pdf(x, a, lg, sg), label='Гамма-распределение')

    plt.title('Аппроксимация распределений для данных')
    plt.legend()
    plt.grid()
    plt.show()

    # Задание 4
    bins = np.histogram_bin_edges(data, bins='auto')
    observed_freq, _ = np.histogram(data, bins=bins)
    expected_freq_norm = np.diff(norm.cdf(bins, mu, std)) * len(data)
    expected_freq_expon = np.diff(expon.cdf(bins, le, se)) * len(data)
    expected_freq_gamma = np.diff(gamma.cdf(bins, a, lg, sg)) * len(data)

    chi2_norm = chi_square_test(observed_freq, expected_freq_norm)
    chi2_expon = chi_square_test(observed_freq, expected_freq_expon)
    chi2_gamma = chi_square_test(observed_freq, expected_freq_gamma)

    print(f"Хи^2 для нормального распределения: {chi2_norm}")
    print(f"Хи^2 для экспоненциального распределения: {chi2_expon}")
    print(f"Хи^2 для гамма-распределения: {chi2_gamma}")

    alpha = 0.05
    df = len(observed_freq) - 1
    critical_value = chi2.ppf(1 - alpha, df)
    print(f"Критическое значение Хи^2 для уровня значимости {alpha}: {critical_value}")

    print(f"Нормальное распределение: {'принимаем' if check_hypothesis(chi2_norm, critical_value) else 'отвергаем'}")
    print(f"Экспоненциальное распределение: {'принимаем' if check_hypothesis(chi2_expon, critical_value) else 'отвергаем'}")
    print(f"Гамма-распределение: {'принимаем' if check_hypothesis(chi2_gamma, critical_value) else 'отвергаем'}")

    # Задание 6
    p_norm = chi2_contingency([observed_freq, expected_freq_norm])
    p_expon = chi2_contingency([observed_freq, expected_freq_expon])
    p_gamma = chi2_contingency([observed_freq, expected_freq_gamma])
    print(f"Хи^2 для нормального распределения (scipy): {p_norm.statistic}")
    print(f"Хи^2 для экспоненциального распределения (scipy): {p_expon.statistic}")
    print(f"Хи^2 для гамма-распределения (scipy): {p_gamma.statistic}")

if __name__ == "__main__":
    main()