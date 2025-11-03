import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")

df = df.head(500)

print("Первые 5 строк датасета (из 500):")
print(df.head())
print()

print("Информация о датасете (500 строк):")
print(df.info())
print()

print("Пропущенные значения в каждом столбце:")
missing_values = df.isnull().sum()
print(missing_values)
print()

print("Заполняем пропущенные значения...")

numeric_columns = df.select_dtypes(include='number').columns

for column in numeric_columns:
    if df[column].isnull().any():
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)
        print(f"Заполнили {column} медианой: {median_value}")

text_columns = df.select_dtypes(include='object').columns

for column in text_columns:
    if df[column].isnull().any():
        mode_values = df[column].mode()
        if not mode_values.empty:
            mode_value = mode_values[0]
        else:
            mode_value = "Unknown"

        df[column].fillna(mode_value, inplace=True)
        print(f"Заполнили {column} значением: '{mode_value}'")

print("\nПроверяем результат заполнения:")
print(df.isnull().sum())
print()

print("Нормализуем числовые данные...")

numeric_cols_for_scaling = df.select_dtypes(include='number').columns

for column in numeric_cols_for_scaling:
    min_val = df[column].min()
    max_val = df[column].max()

    if max_val != min_val:
        df[column] = (df[column] - min_val) / (max_val - min_val)
        print(f"Нормализовали {column} в диапазон [0, 1]")
    else:
        print(f"Столбец {column} имеет постоянное значение, нормализация не нужна")

print("Нормализация завершена")
print()

print("Преобразуем категориальные данные...")

categorical_columns = df.select_dtypes(include='object').columns

if len(categorical_columns) > 0:
    print(f"Преобразуем столбцы: {list(categorical_columns)}")

    for column in categorical_columns:
        if df[column].nunique() > 10:
            freq_encoding = df[column].value_counts().to_dict()
            df[column + '_freq'] = df[column].map(freq_encoding)
            print(f"Применили частотное кодирование для {column} (уникальных значений: {df[column].nunique()})")
        else:
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            df[column + '_label'] = le.fit_transform(df[column])
            print(f"Применили Label Encoding для {column} (уникальных значений: {df[column].nunique()})")

    df.drop(columns=categorical_columns, inplace=True)
    print("Оптимизированное преобразование категориальных данных применено успешно")
else:
    print("Категориальных столбцов для преобразования нет")

print("Преобразование категориальных данных завершено")
print()

print("Удаляем ненужные столбцы...")

columns_to_remove = [
    'CryoSleep', 'VIP', 'Transported', 'PassengerId_freq',
    'HomePlanet_label', 'Cabin_freq', 'Destination_label', 'Name_freq'
]

existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]

if existing_columns_to_remove:
    df.drop(columns=existing_columns_to_remove, inplace=True)
    print(f"Удалены столбцы: {existing_columns_to_remove}")
else:
    print("Ненужные столбцы не найдены в датафрейме")

print("Итоговые данные после предобработки:")
print(df.head())
print(f"\nРазмер данных после обработки: {df.shape}")

df.to_csv("processed_data.csv", index=False, encoding='utf-8-sig')
print("\nОбработанные данные сохранены в файл 'processed_data.csv'")

print("\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
print(f"Исходный размер данных: информация из метода info() выше")
print(f"Конечный размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"Типы данных после обработки:")
print(df.dtypes.value_counts())

print(f"\nНазвания столбцов после обработки ({len(df.columns)} шт.):")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

import os

if os.path.exists("processed_data.csv"):
    file_size = os.path.getsize("processed_data.csv") / (1024 * 1024)
    print(f"\nРазмер файла: {file_size:.2f} МБ")