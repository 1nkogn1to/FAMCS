import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import folium
from numpy import pi


def radial_plot(dataframe):
    df = dataframe.groupby('owner').size().reset_index(name='counts')

    categories = list(df['owner'])
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories)
    values = df['counts'].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)
    plt.title('Количество машин по типу владельца')
    plt.show()


def plot_different_graphics(dataframe):
    plt.figure(figsize=(10, 6))
    sb.countplot(x='transmission', data=dataframe)
    plt.title('Количество автомобилей по типу трансмиссии')
    plt.xlabel('Тип топлива')
    plt.ylabel('Количество')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sb.histplot(dataframe['selling_price'], bins=30, kde=True)
    plt.title('Распределение цены автомобилей')
    plt.xlabel('Цена продажи')
    plt.ylabel('Частота')
    plt.show()

    plt.figure(figsize=(10, 6))
    sb.lineplot(x='owner', y='selling_price', data=dataframe)
    plt.title('Цена продажи в зависимости от владельца')
    plt.xlabel('Владельцы')
    plt.ylabel('Цена продажи')
    plt.show()

    plt.figure(figsize=(10, 6))
    sb.boxplot(x='seller_type', y='selling_price', data=dataframe)
    plt.title('Цена продажи у разных продавцов')
    plt.xlabel('Продавец')
    plt.ylabel('Цена продажи')
    plt.show()

    radial_plot(dataframe)

    plt.figure(figsize=(10, 6))
    sb.scatterplot(x='km_driven', y='selling_price', hue='fuel', data=dataframe)
    plt.title('Цена продажи в зависимости от пробега')
    plt.xlabel('Пробег')
    plt.ylabel('Цена продажи')
    plt.show()

def plot_temperature_map(dataframe):
    numeric_df = dataframe.select_dtypes(include=[float, int])
    corrMatr = numeric_df.corr()
    plt.figure(figsize=(20, 8))
    sb.heatmap(corrMatr, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Корреляция между параметрами погоды')
    plt.show()

def plot_interactive(dataframe):
    data = dataframe[['State', 'Total Ballots Counted (Estimate)', 'VEP Turnout Rate', 'Voting-Eligible Population (VEP)', 'State Abv']]
    data = data.dropna()  # Удаляем строки с пропущенными значениями

    fig = px.choropleth(
        data_frame=data,
        locations="State Abv",  # Используем сокращения штатов для отображения
        locationmode='USA-states',  # Режим отображения штатов США
        color="VEP Turnout Rate",  # Цветовая шкала на основе процента явки
        hover_name="State",  # Название штата при наведении
        hover_data={"Voting-Eligible Population (VEP)", "Total Ballots Counted (Estimate)", "VEP Turnout Rate"},  # Дополнительная информация при наведении
        scope="usa",  # Масштаб карты (США)
        title="Явка на выборах в США 2020 по штатам"
    )
    fig.update_layout()
    fig.show()

    fig_bar = px.bar(
        data_frame=data,
        x="State",
        y="VEP Turnout Rate",
        hover_data=["Voting-Eligible Population (VEP)", "Total Ballots Counted (Estimate)"],
        title="Процент явки на выборах по штатам",
        labels={"VEP Turnout Rate": "Процент явки (%)"},
        color="VEP Turnout Rate",
    )

    fig_bar.update_layout(xaxis={'categoryorder': 'total descending'})  # Сортировка по убыванию процента явки
    fig_bar.show()

def plot_covid_info(dataframe_c, dataframe_d, dataframe_r):
    confGlobal = dataframe_c.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long']).sum()
    deathGlobal = dataframe_d.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long']).sum()
    recGlobal = dataframe_r.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long']).sum()

    confGlobal.index = pd.to_datetime(confGlobal.index)
    deathGlobal.index = pd.to_datetime(deathGlobal.index)
    recGlobal.index = pd.to_datetime(recGlobal.index)

    plt.figure(figsize=(14, 7))

    plt.plot(confGlobal, label='Подтвержденные случаи')
    plt.plot(deathGlobal, label='Смерти')
    plt.plot(recGlobal, label='Выздоровления')

    plt.xlabel('Дата')
    plt.ylabel('Количество случаев')
    plt.title('Изменение числа случаев заражения, смертей и выздоровлений по времени')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_map_of_terror_attacks(dataframe):
    df_filtered = dataframe[dataframe['success'] == 1]

    df_filtered = df_filtered.dropna(subset=['latitude', 'longitude'])

    map_terror = folium.Map(location=[20, 0], zoom_start=2)

    for idx, row in df_filtered.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            popup=f"City: {row['city']}<br>Country: {row['country_txt']}<br>Year: {row['iyear']}<br>Attack Type: {row['attacktype1_txt']}",
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(map_terror)

    map_terror.save('terrorist_attacks_map.html')

def main():
    # Задание 1
    df_task1 = pd.read_csv('info/CAR DETAILS FROM CAR DEKHO.csv')
    plot_different_graphics(df_task1)

    # Задание 2
    df_task2 = pd.read_csv('info/temperature.csv')
    plot_temperature_map(df_task2)

    # Задание 3
    df_task3 = pd.read_csv('info/2020 November General Election - Turnout Rates.csv')
    plot_interactive(df_task3)

    # Задание 4
    df_task4_c = pd.read_csv('info/time_series_covid19_confirmed_global.csv')
    df_task4_d = pd.read_csv('info/time_series_covid19_deaths_global.csv')
    df_task4_r = pd.read_csv('info/time_series_covid19_recovered_global.csv')
    plot_covid_info(df_task4_c, df_task4_d, df_task4_r)


    # Задание 5
    #df_task5 = pd.read_csv('info/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
    #plot_map_of_terror_attacks(df_task5)


if __name__ == "__main__":
    main()