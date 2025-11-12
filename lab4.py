import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('processed_data.csv')

print("Распределение целевой переменной:")
target_counts = df.iloc[:, -1].value_counts().sort_index()
print(target_counts)
print(f"Всего классов: {df.iloc[:, -1].nunique()}")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
n_classes = len(class_names)

print(f"Закодированные классы: {dict(zip(class_names, range(len(class_names))))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nРазмеры данных:")
print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")


rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)
rf_oob_score = rf.oob_score_

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)


def safe_evaluation(model, model_name, X_test, y_test, class_names):
    print(f"\n{model_name} - Результаты:")

    y_pred = model.predict(X_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(
            y_test, y_pred,
            target_names=[str(cls) for cls in class_names],
            zero_division=0
        )
        print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("Матрица ошибок:")
    print(cm)

    return y_pred

y_pred_rf = safe_evaluation(rf, "СЛУЧАЙНЫЙ ЛЕС", X_test, y_test, class_names)
y_pred_ada = safe_evaluation(ada, "ADABOOST", X_test, y_test, class_names)
y_pred_gb = safe_evaluation(gb, "ГРАДИЕНТНЫЙ БУСТИНГ", X_test, y_test, class_names)

print(f"\nRandom Forest OOB Score: {rf_oob_score:.4f}")

print(f"\nПостроение ROC-кривых для {n_classes} классов...")

y_test_bin = label_binarize(y_test, classes=range(n_classes))

y_proba_rf = rf.predict_proba(X_test)
y_proba_ada = ada.predict_proba(X_test)
y_proba_gb = gb.predict_proba(X_test)

fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()

fpr_ada = dict()
tpr_ada = dict()
roc_auc_ada = dict()

fpr_gb = dict()
tpr_gb = dict()
roc_auc_gb = dict()

for i in range(n_classes):
    # Random Forest
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], y_proba_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

    # AdaBoost
    fpr_ada[i], tpr_ada[i], _ = roc_curve(y_test_bin[:, i], y_proba_ada[:, i])
    roc_auc_ada[i] = auc(fpr_ada[i], tpr_ada[i])

    # Gradient Boosting
    fpr_gb[i], tpr_gb[i], _ = roc_curve(y_test_bin[:, i], y_proba_gb[:, i])
    roc_auc_gb[i] = auc(fpr_gb[i], tpr_gb[i])

all_fpr_rf = np.unique(np.concatenate([fpr_rf[i] for i in range(n_classes)]))
all_fpr_ada = np.unique(np.concatenate([fpr_ada[i] for i in range(n_classes)]))
all_fpr_gb = np.unique(np.concatenate([fpr_gb[i] for i in range(n_classes)]))

mean_tpr_rf = np.zeros_like(all_fpr_rf)
mean_tpr_ada = np.zeros_like(all_fpr_ada)
mean_tpr_gb = np.zeros_like(all_fpr_gb)

for i in range(n_classes):
    mean_tpr_rf += np.interp(all_fpr_rf, fpr_rf[i], tpr_rf[i])
    mean_tpr_ada += np.interp(all_fpr_ada, fpr_ada[i], tpr_ada[i])
    mean_tpr_gb += np.interp(all_fpr_gb, fpr_gb[i], tpr_gb[i])

mean_tpr_rf /= n_classes
mean_tpr_ada /= n_classes
mean_tpr_gb /= n_classes

roc_auc_rf_macro = auc(all_fpr_rf, mean_tpr_rf)
roc_auc_ada_macro = auc(all_fpr_ada, mean_tpr_ada)
roc_auc_gb_macro = auc(all_fpr_gb, mean_tpr_gb)

plt.figure(figsize=(12, 8))

plt.plot(all_fpr_rf, mean_tpr_rf,
         label=f'Random Forest (macro-average AUC = {roc_auc_rf_macro:.3f})',
         color='blue', linewidth=3)
plt.plot(all_fpr_ada, mean_tpr_ada,
         label=f'AdaBoost (macro-average AUC = {roc_auc_ada_macro:.3f})',
         color='red', linewidth=3)
plt.plot(all_fpr_gb, mean_tpr_gb,
         label=f'Gradient Boosting (macro-average AUC = {roc_auc_gb_macro:.3f})',
         color='green', linewidth=3)

colors = ['lightblue', 'lightcoral', 'lightgreen', 'yellow', 'orange', 'pink']
for i in range(min(n_classes, 6)):  # Показываем максимум 6 классов для читаемости
    plt.plot(fpr_rf[i], tpr_rf[i], color=colors[i], linestyle=':',
             label=f'RF Class {class_names[i]} (AUC = {roc_auc_rf[i]:.3f})', alpha=0.6)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC-кривые (многоклассовая классификация, {n_classes} классов)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nMACRO-AVERAGE AUC SCORES:")
print(f"Random Forest: {roc_auc_rf_macro:.4f}")
print(f"AdaBoost: {roc_auc_ada_macro:.4f}")
print(f"Gradient Boosting: {roc_auc_gb_macro:.4f}")

print(f"\nAUC по классам для Random Forest:")
for i in range(n_classes):
    print(f"  Класс {class_names[i]}: {roc_auc_rf[i]:.4f}")

from sklearn.metrics import accuracy_score

rf_accuracy = accuracy_score(y_test, y_pred_rf)
ada_accuracy = accuracy_score(y_test, y_pred_ada)
gb_accuracy = accuracy_score(y_test, y_pred_gb)

print("СРАВНЕНИЕ МОДЕЛЕЙ")
print(f"Random Forest Accuracy: {rf_accuracy:.4f} (OOB: {rf_oob_score:.4f})")
print(f"AdaBoost Accuracy: {ada_accuracy:.4f}")
print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")

accuracies = {
    'Random Forest': rf_accuracy,
    'AdaBoost': ada_accuracy,
    'Gradient Boosting': gb_accuracy
}
best_model = max(accuracies, key=accuracies.get)
print(f"\nЛучшая модель по точности: {best_model} ({accuracies[best_model]:.4f})")