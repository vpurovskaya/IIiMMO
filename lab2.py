import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

print("\n1. ЗАГРУЗКА ДАННЫХ")
try:
    df = pd.read_csv("processed_data.csv")
    print("Данные загружены успешно")
    print(f"Размер данных: {df.shape}")
except FileNotFoundError:
    print("Файл processed_data.csv не найден!")
    exit()

print("\n2. АНАЛИЗ СТРУКТУРЫ ДАННЫХ")

numeric_columns = df.select_dtypes(include=[np.number]).columns
print(f"Найдено числовых колонок: {len(numeric_columns)}")

potential_targets = [col for col in numeric_columns if df[col].nunique() > 2]
if potential_targets:
    regression_target = potential_targets[0]
else:
    regression_target = numeric_columns[0]

print(f"Целевая переменная для регрессии: '{regression_target}'")

binary_columns = [col for col in numeric_columns if df[col].nunique() == 2]
if binary_columns:
    classification_target = binary_columns[0]
    print(f"Целевая переменная для классификации: '{classification_target}'")
else:
    classification_target = 'Age_Category'
    median_age = df[regression_target].median()
    df[classification_target] = (df[regression_target] > median_age).astype(int)
    print(f"Создана целевая переменная для классификации: '{classification_target}'")

print("\n3. РАЗДЕЛЕНИЕ ДАННЫХ")

features = [col for col in df.columns if col not in [regression_target, classification_target]]
X = df[features]

y_reg = df[regression_target]
y_clf = df[classification_target]

print(f"Признаки: {X.shape[1]} колонок")
print(f"Наблюдения: {X.shape[0]} строк")

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

print(f"Обучающая выборка: {X_train_reg.shape[0]} наблюдений")
print(f"Тестовая выборка: {X_test_reg.shape[0]} наблюдений")

print(f"\n4. ЗАДАЧА РЕГРЕССИИ (предсказание '{regression_target}')")

linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)

y_pred_reg = linear_model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = linear_model.score(X_test_reg, y_test_reg)

print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Коэффициент детерминации (R²): {r2:.4f}")

print(f"\n5. ЗАДАЧА КЛАССИФИКАЦИИ (предсказание '{classification_target}')")

print("Распределение классов:")
class_counts = y_clf.value_counts()
for class_val, count in class_counts.items():
    percentage = count / len(y_clf) * 100
    print(f"  Класс {class_val}: {count} наблюдений ({percentage:.1f}%)")

logreg_model = LogisticRegression(random_state=42, max_iter=1000)
logreg_model.fit(X_train_clf, y_train_clf)

y_pred_clf = logreg_model.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
cm = confusion_matrix(y_test_clf, y_pred_clf)

print(f"Точность (Accuracy): {accuracy:.4f}")
print("Матрица ошибок:")
print(cm)

print(f"\nРЕЗУЛЬТАТЫ:")
print(f"  Регрессия - R²: {r2:.4f}")
print(f"  Классификация - Accuracy: {accuracy:.4f}")

print(f"\nРЕКОМЕНДАЦИИ:")
if r2 < 0.3:
    print("Регрессия: низкое качество")
    print("   - Попробуйте добавить полиномиальные признаки")
    print("   - Используйте регуляризацию (Ridge, Lasso)")
elif r2 < 0.6:
    print("Регрессия: среднее качество")
    print("   - Можно улучшить подбором гиперпараметров")
else:
    print("Регрессия: хорошее качество")

if accuracy < 0.7:
    print("Классификация: низкая точность")
    print("   - Попробуйте балансировку классов")
    print("   - Настройте гиперпараметры модели")
elif accuracy < 0.85:
    print("Классификация: средняя точность")
    print("   - Можно улучшить подбором порога классификации")
else:
    print("Классификация: высокая точность")
