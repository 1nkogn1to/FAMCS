import pandas as pd
from datetime import datetime, timedelta


def percentage_of_null_data(dataframe, column):
    missing_values = dataframe[column].isnull().sum()
    return missing_values / len(dataframe[column]) * 100

def check(data, format_):
    try:
        pd.to_datetime(data, format=format_)
        return 1
    except ValueError:
        return 0
    
def consistency_of_data_dtformat(dataframe, number_of_lines, column, format_):
    count = 0
    for _ in range(number_of_lines):
        count += check(dataframe[column][_], format_)

    return count

def result_of_analize_of_date(dataframe):
    how_much_to_check = 1000
    correct_format = consistency_of_data_dtformat(dataframe, how_much_to_check, 'dt', '%Y-%m-%d')
    wrong_format = how_much_to_check - correct_format
    if wrong_format == 0:
        print("Date format is correct")
    else:
        print(f"Date format is incorrect in {wrong_format} of {how_much_to_check} lines")

def topical_tweets(dataframe):
    data = pd.to_datetime(dataframe['date'], format='%Y-%m-%d %H:%M:%S')
    correct = consistency_of_data_dtformat(dataframe, len(data), 'date', '%Y-%m-%d %H:%M:%S')
    if (correct == len(data)):
        some_time_ago = datetime.now() - timedelta(days=1550)
        topical_info = data[data >= some_time_ago]
        print(f"Количество актуальных твитов (не более 1550 дней назад) - {len(topical_info)},\nпроцент актуальных твитов по отношению ко всем в выборке - {len(topical_info) / len(data) * 100}")

def cancer_statistics(dataframe):
    uc = dataframe['Country'].nunique()
    ac = 185
    mc = ac - uc
    print(f"По нескольким странам не собрана информация, а именно по {mc}")

def main():
    #df_task1 = pd.read_csv("info/online_retail_II.csv", encoding='ISO-8859-1')
    #print(percentage_of_null_data(df_task1, "Customer ID"))

    df_task2 = pd.read_csv("info/GlobalLandTemperaturesByCity.csv")
    result_of_analize_of_date(df_task2)

    df_task3 = pd.read_csv('info/covid19_tweets.csv')
    topical_tweets(df_task3)
    
    #df_task4 = pd.read_csv('info/GLOBOCAN_2018.csv')
    #cancer_statistics(df_task4)


if __name__ == "__main__":
    main()