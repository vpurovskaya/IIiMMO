import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('processed_data.csv')

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"1. РАЗДЕЛЕНИЕ ДАННЫХ:")
print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")

print("\n2. ЗАДАЧА РЕГРЕССИИ - ПРЕДСКАЗАНИЕ SalePrice")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg_model.fit(X_train, y_train)
y_pred_rf = rf_reg_model.predict(X_test)

print("\n3. ОЦЕНКА РЕГРЕССИОННЫХ МОДЕЛЕЙ:")

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Линейная регрессия - MSE: {mse_lr:.4f}, R²: {r2_lr:.4f}")
print(f"Случайный лес - MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title(f'Линейная регрессия (R² = {r2_lr:.4f})')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title(f'Случайный лес (R² = {r2_rf:.4f})')

plt.tight_layout()
plt.show()

print("\nАНАЛИЗ РЕГРЕССИИ:")
if r2_rf < 0.7:
    print("Результаты можно улучшить:")
    print("- Подбор гиперпараметров моделей")
    print("- Использование градиентного бустинга")
    print("- Удаление мультиколлинеарных признаков")
else:
    print("Хорошие результаты регрессии!")

print("\n4. ЗАДАЧА КЛАССИФИКАЦИИ")

price_median = df['SalePrice'].median()
y_class = (df['SalePrice'] > price_median).astype(int)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

logreg_model = LogisticRegression(random_state=42, max_iter=1000)
logreg_model.fit(X_train_clf, y_train_clf)
y_pred_logreg = logreg_model.predict(X_test_clf)

rf_clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf_model.fit(X_train_clf, y_train_clf)
y_pred_rf_clf = rf_clf_model.predict(X_test_clf)

print("\n5. ОЦЕНКА КЛАССИФИКАЦИОННЫХ МОДЕЛЕЙ:")

accuracy_logreg = accuracy_score(y_test_clf, y_pred_logreg)

accuracy_rf = accuracy_score(y_test_clf, y_pred_rf_clf)

print(f"Логистическая регрессия - Accuracy: {accuracy_logreg:.4f}")
print(f"Случайный лес - Accuracy: {accuracy_rf:.4f}")

print("\nОТЧЕТ ПО КЛАССИФИКАЦИИ (Случайный лес):")
print(classification_report(y_test_clf, y_pred_rf_clf))

print("\nАНАЛИЗ КЛАССИФИКАЦИИ:")
if accuracy_rf < 0.8:
    print("Результаты можно улучшить:")
    print("- Балансировка классов")
    print("- Подбор порога классификации")
    print("- Использование ансамблевых методов")
else:
    print("Хорошие результаты классификации!")