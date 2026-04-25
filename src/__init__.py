import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('B:/ai_prjct/hseml-group-project-minecraftteam/data/room_dataset.csv')

def classify_repair(score):
    if score <= 0.04:
        return 0
    elif score <= 0.08:
        return 1
    else:
        return 2

df['repair_type'] = df['defect_score'].apply(classify_repair)

feature_cols = [col for col in df.columns if col not in 
                ['filename', 'defect_score', 'repair_type']]

X = df[feature_cols].copy()
y = df['repair_type']

print("Распределение классов до обработки:")
print(y.value_counts())

min_count = y.value_counts().min()
if min_count < 2:
    print(f"\nОбъединяем класс с {min_count} объектом(ми)")
    rare_class = y.value_counts().idxmin()
    if rare_class == 0:
        y = y.replace({0: 1})
    elif rare_class == 2:
        y = y.replace({2: 1})
    else:
        y = y.replace({1: 0})
    print("Новое распределение:")
    print(y.value_counts())

numeric_features = ['brightness', 'contrast', 'blur_score', 'edge_density',
                    'num_objects', 'green_ratio', 'color_entropy',
                    'wall_floor_ratio', 'light_uniformity', 'num_windows',
                    'furniture_count', 'furniture_density', 'aesthetic_score']

categorical_features = ['location', 'scene_category']

numeric_features = [f for f in numeric_features if f in X.columns]
categorical_features = [f for f in categorical_features if f in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\nBaseline Logistic Regression")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")

unique_classes = sorted(y.unique())
class_names = {0: 'Без ремонта', 1: 'Косметический', 2: 'Капитальный'}
target_names = [class_names[c] for c in unique_classes]

print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_classes))

def predict_repair_from_row(row):
    input_df = pd.DataFrame([row])[feature_cols]
    pred = pipeline.predict(input_df)[0]
    return class_names[pred]

print("\nВЕРДИКТЫ ДЛЯ ВСЕХ ОБЪЕКТОВ:")

df['predicted_repair'] = df[feature_cols].apply(predict_repair_from_row, axis=1)

for idx, row in df.iterrows():
    print(f"{row['filename']:35s} | Истинный: {class_names[row['repair_type']]:15s} | Предсказанный: {row['predicted_repair']}")

print("\nСВОДНАЯ СТАТИСТИКА:")
print(df['predicted_repair'].value_counts())

df[['filename', 'repair_type', 'predicted_repair']].to_csv('B:/ai_prjct/hseml-group-project-minecraftteam/data/repair_verdicts.csv', index=False)
print("\nРезультаты сохранены в файл 'repair_verdicts.csv'")

if len(numeric_features) > 0 and len(unique_classes) > 1:
    X_num = X[numeric_features]
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    model_num = LogisticRegression(max_iter=1000)
    model_num.fit(X_num_scaled, y)
    coeffs = pd.DataFrame({
        'признак': numeric_features,
        'коэффициент': abs(model_num.coef_[0])
    }).sort_values('коэффициент', ascending=False)
    print("\nТоп-5 признаков, влияющих на ремонт:")
    print(coeffs.head())