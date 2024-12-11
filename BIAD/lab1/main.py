import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
import scipy.stats as stats

df = pd.read_csv('info/avianHabitat.csv')

Adf = df[df["AHt"] != 0]
Edf = df[df["EHt"] != 0]

mx = max(Adf["AHt"])
mn = min(Adf["AHt"])
print(len(Adf["AHt"]))

print(f"Максимальное и минмальное значения - {mx}, {mn}")
print(f"Разброс распределения - {mx - mn}")
avg = Adf["AHt"].mean()
print(f"Среднее значение - {avg}")
print(f"Медиана - {Adf["AHt"].median()}")
print(f"Дисперсия - {Adf["AHt"].var()}")
std = Adf["AHt"].std()
print(f"Среднеквадратичное отклонение - {std}")

q1 = Adf["AHt"].quantile(0.25)
q3 = Adf["AHt"].quantile(0.75)
print(f"Первый и третий квартили - {q1}, {q3}")
Iqr = q3 - q1
print(f"Интерквартильный размах - {Iqr}")

n = len(Adf["AHt"])
print(f"Ассиметрия - {n / ((n - 1) * (n - 2)) * np.sum(((Adf["AHt"] - avg) / std)**3)}")
print(f"Ассиметрия встроенным методом - {Adf["AHt"].skew()}")

print(f"Эксцесс - {(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(((Adf["AHt"] - avg) / std)**4) - (3 * (n - 1)**2) / ((n - 2) * (n - 3))}")
print(f"Эксцесс встроенным методом - {Adf["AHt"].kurtosis()}")

plt.figure(figsize=(10, 6))
s.boxplot(y=Adf["AHt"])
plt.show()

plt.figure(figsize=(12, 6))
s.boxplot(data=(Adf["AHt"], Edf["EHt"]))
plt.show()

data_sorted = Adf['AHt'].sort_values().values
n = len(data_sorted)
y_values = np.arange(1, n + 1) / n

plt.step(data_sorted, y_values, where='post', color='blue')

plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
s.histplot(Adf['AHt'], bins=30, kde=True, stat="probability")

plt.xlabel('Значение AHt')
plt.ylabel('Вероятность')
plt.grid()
plt.show()

data = Adf['AHt']

stats.probplot(data, dist="norm", plot=plt)
plt.grid()
plt.show()