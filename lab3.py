import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('processed_data.csv')

print("Посмотрим на данные:")
print(df.head())
print("\nРазмер данных:", df.shape)
print("\nТипы данных:")
print(df.dtypes)

if df.isnull().sum().sum() > 0:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

print("\nПосле обработки:")
print(df.head())

print("Регрессия")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

target_col = None
for col in numeric_cols:
    if df[col].nunique() > 10:
        target_col = col
        break

if target_col is None:
    target_col = numeric_cols[0]

print(f"Целевая переменная для регрессии: {target_col}")

feature_cols = [col for col in numeric_cols if col != target_col]
print(f"Признаки: {feature_cols}")

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nОшибка MSE: {mse:.4f}")
print(f"R² score: {r2:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Настоящие значения')
plt.ylabel('Предсказания')
plt.title('Регрессия: настоящие vs предсказанные')

plt.subplot(1, 2, 2)
plt.bar(['MSE', 'R²'], [mse, r2], color=['lightcoral', 'lightblue'])
plt.title('Метрики регрессии')
plt.ylabel('Значение')

plt.tight_layout()
plt.show()

print("Классификация И ROC-Кривая")


binary_target = None
for col in numeric_cols:
    if df[col].nunique() == 2:
        binary_target = col
        break

if binary_target is None:
    first_col = numeric_cols[0]
    median_val = df[first_col].median()
    df['binary_target'] = (df[first_col] > median_val).astype(int)
    binary_target = 'binary_target'
    print(f"Создана бинарная цель: {binary_target}")

print(f"Целевая переменная для классификации: {binary_target}")

class_features = [col for col in numeric_cols if col != binary_target]

X_clf = df[class_features]
y_clf = df[binary_target]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_clf, y_train_clf)

y_proba = tree_clf.predict_proba(X_test_clf)

fpr, tpr, thresholds = roc_curve(y_test_clf, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

print(f"\nПлощадь под ROC-кривой (AUC): {roc_auc:.4f}")

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайная модель')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.grid(True)

for i in range(0, len(thresholds), 10):
    plt.plot(fpr[i], tpr[i], 'o', markersize=4, color='red', alpha=0.6)

plt.show()

print("\nПримеры порогов и соответствующих TPR/FPR:")
for i in range(0, len(thresholds), 20):
    if i < len(thresholds):
        print(f"Порог: {thresholds[i]:.3f} -> TPR: {tpr[i]:.3f}, FPR: {fpr[i]:.3f}")
