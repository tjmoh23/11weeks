# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['target'] = wine.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#  HyperParameter Tunning
param_grid = {
    "criterion" : ['gini', 'entropy'],
    "max_depth" : [2,3,4,5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# DT 모델 생성 및 그리드서치
dt_grid = DecisionTreeClassifier(random_state= 42)
df_grid_search = GridSearchCV(dt_grid, param_grid, cv = 5)
df_grid_search.fit(X_train, y_train)

print("Best Hyper-parameter", df_grid_search.best_params_)
print("Best Score", df_grid_search.best_score_)


# 정확도 계산 
dt_best_model = df_grid_search.best_estimator_

dt_y_pred_grid = dt_best_model.predict(X_test)
dt_accuracy_grid = accuracy_score(y_test, dt_y_pred_grid)
# print('Accuracy Grid :', accuracy_grid)


# Feature Importance를 계산
dt_importances = dt_best_model.feature_importances_

# Best model의 Feature Importance를 시각화
plt.figure(figsize = (20,6))

# 막대 그래프 생성
plt.bar(range(len(dt_importances)), dt_importances, width=0.3)
plt.xlabel('Feature')
plt.ylabel('importances')
plt.title('Feature Importance')
plt.xticks(range(len(dt_importances)), X.columns, rotation = 45)
plt.show()


####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

