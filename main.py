import pandas as pd

df = pd.read_csv("train.csv")

print("Первые 5 строк датасета:")
print(df.head())
print()

print("Информация о датасете:")
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

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    print("One-Hot Encoding применен успешно")
else:
    print("Категориальных столбцов для преобразования нет")

print("Преобразование категориальных данных завершено")
print()

print("Итоговые данные после предобработки:")
print(df.head())
print(f"\nРазмер данных после обработки: {df.shape}")

df.to_csv("processed_data.csv", index=False)
print("\nОбработанные данные сохранены в файл 'processed_data.csv'")

print("\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
print(f"Исходный размер данных: информация из метода info() выше")
print(f"Конечный размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"Типы данных после обработки:")
print(df.dtypes.value_counts())

print(f"\nНазвания столбцов после обработки ({len(df.columns)} шт.):")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")