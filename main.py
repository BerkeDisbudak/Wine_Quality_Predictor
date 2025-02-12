import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


wine_train = pd.read_csv('./winequality-red.csv', sep=';')
wine_test = pd.read_csv('./winequality-red.csv', sep=';')


print(wine_train.columns)
print(wine_train.isna().sum())
print(wine_train.isnull().sum())

#correlation heatmap
corr = wine_train.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

plt.figure(figsize=(12,12))
sns.boxplot(data=wine_train)
plt.xticks(rotation=90)
plt.show()

#IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), outliers

# Tüm sayısal değişkenlerde uç değer sayısını görmek için
outlier_counts = {}
for col in wine_train.select_dtypes(include=[np.number]).columns:
    count, _ = detect_outliers_iqr(wine_train, col)
    outlier_counts[col] = count

sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
for feature, count in sorted_outliers:
    print(f"{feature}: {count} Outliers:")

#Clipping
columns_to_clip = ["residual sugar", "chlorides", "sulphates", "density", "pH", "fixed acidity"]
for col in columns_to_clip:
    lower_bound = wine_train[col].quantile(0.01)  # %1 alt sınır
    upper_bound = wine_train[col].quantile(0.99)  # %99 üst sınır
    wine_train[col] = np.clip(wine_train[col], lower_bound, upper_bound)
    wine_test[col] = np.clip(wine_test[col], lower_bound, upper_bound)  # Test setinde de uyguluyoruz


wine_train['total_acidity'] = wine_train['fixed acidity'] + wine_train['volatile acidity']
wine_train['sulfur_dioxide_ratio'] = wine_train['free sulfur dioxide'] / wine_train['total sulfur dioxide']
wine_train['alcohol_index'] = wine_train['alcohol'] / wine_train['density']

wine_test['total_acidity'] = wine_test['fixed acidity'] + wine_test['volatile acidity']
wine_test['sulfur_dioxide_ratio'] = wine_test['free sulfur dioxide'] / wine_test['total sulfur dioxide']
wine_test['alcohol_index'] = wine_test['alcohol'] / wine_test['density']

#Log Transform
wine_train['total sulfur dioxide'] = np.log1p(wine_train['total sulfur dioxide'])
wine_train['free sulfur dioxide'] = np.log1p(wine_train['free sulfur dioxide'])
wine_train['residual sugar'] = np.log1p(wine_train['residual sugar'])
wine_train['total_acidity'] = np.log1p(wine_train['total_acidity'])
wine_train['chlorides'] = np.log1p(wine_train['chlorides'])
wine_train['fixed acidity'] = np.log1p(wine_train['fixed acidity'])

wine_test['total sulfur dioxide'] = np.log1p(wine_test['total sulfur dioxide'])
wine_test['free sulfur dioxide'] = np.log1p(wine_test['free sulfur dioxide'])
wine_test['residual sugar'] = np.log1p(wine_test['residual sugar'])
wine_test['total_acidity'] = np.log1p(wine_test['total_acidity'])
wine_test['chlorides'] = np.log1p(wine_test['chlorides'])
wine_test['fixed acidity'] = np.log1p(wine_test['fixed acidity'])

#Creating new features for better scores.
wine_train['acid_difference'] = wine_train['fixed acidity'] - wine_train['volatile acidity']
wine_train['acid_ratio'] = wine_train['fixed acidity'] / (wine_train['volatile acidity'] + 1e-5)  # 1e-5, sıfır bölme hatasını engeller
wine_train['acid_pH_interaction'] = wine_train['fixed acidity'] * wine_train['pH']
wine_train['alcohol_acid_interaction'] = wine_train['alcohol'] * (wine_train['fixed acidity'] + wine_train['volatile acidity'])
wine_train['alcohol_citric_interaction'] = wine_train['alcohol'] * wine_train['citric acid']
wine_train['chlorides_sulfur_interaction'] = wine_train['chlorides'] * wine_train['total sulfur dioxide']
wine_train['log_residual_sugar'] = np.log(wine_train['residual sugar'] + 1)  # Log dönüşümü
wine_train['pH_density_ratio'] = wine_train['pH'] / (wine_train['density'] + 1e-5)  # 1e-5 sıfır bölmeyi engeller
wine_train['sulphates_pH_difference'] = wine_train['sulphates'] - wine_train['pH']
wine_train['alcohol_sulphates_interaction'] = wine_train['alcohol'] * wine_train['sulphates']
wine_train['quality_alcohol_sulphates_interaction'] = wine_train['quality'] * (wine_train['alcohol'] + wine_train['sulphates'])


wine_test['acid_difference'] = wine_test['fixed acidity'] - wine_test['volatile acidity']
wine_test['acid_ratio'] = wine_test['fixed acidity'] / (wine_test['volatile acidity'] + 1e-5)  # 1e-5, sıfır bölme hatasını engeller
wine_test['acid_pH_interaction'] = wine_test['fixed acidity'] * wine_test['pH']
wine_test['alcohol_acid_interaction'] = wine_test['alcohol'] * (wine_test['fixed acidity'] + wine_test['volatile acidity'])
wine_test['alcohol_citric_interaction'] = wine_test['alcohol'] * wine_test['citric acid']
wine_test['chlorides_sulfur_interaction'] = wine_test['chlorides'] * wine_test['total sulfur dioxide']
wine_test['log_residual_sugar'] = np.log(wine_test['residual sugar'] + 1)  # Log dönüşümü
wine_test['pH_density_ratio'] = wine_test['pH'] / (wine_test['density'] + 1e-5)  # 1e-5 sıfır bölmeyi engeller
wine_test['sulphates_pH_difference'] = wine_test['sulphates'] - wine_test['pH']
wine_test['alcohol_sulphates_interaction'] = wine_test['alcohol'] * wine_test['sulphates']
wine_test['quality_alcohol_sulphates_interaction'] = wine_test['quality'] * (wine_test['alcohol'] + wine_test['sulphates'])

# set dependent and independent features.
X = wine_train.drop("quality", axis=1)
y = wine_train["quality"]

#split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# params for GridSearch
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 7, 10],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}


xgb = XGBRegressor()


grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

#train the model
grid_search.fit(X_train, y_train)

# write down to best parameters we found.
print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

#Model Evaluation
predictions = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("RMSE: ", rmse)
print("R2: ", r2)

results = grid_search.cv_results_

param_names = list(grid_search.best_params_.keys())
param_values = list(grid_search.best_params_.values())

plt.figure(figsize=(12, 8))
plt.barh(param_names, param_values, color='skyblue')
plt.xlabel('Parameter Value')
plt.title('Best Trial Parameters from GridSearchCV')
plt.show()

mean_test_scores = results['mean_test_score']
plt.figure(figsize=(12, 8))
plt.plot(mean_test_scores, label="Mean Test Score (Negative MSE)")
plt.xlabel('Trial Number')
plt.ylabel('Negative MSE')
plt.title('GridSearchCV Performance (Negative MSE)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
sns.residplot(x=predictions, y=y_test - predictions, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Gerçek ve tahmin edilen değerleri karşılaştırmak için scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
plt.show()

# Hataların dağılımını görmek için histogram/kde plot
plt.figure(figsize=(8, 5))
sns.histplot(predictions - y_test, kde=True, bins=30)
plt.xlabel("Prediction Error")
plt.title("Distribution of Prediction Errors")
plt.show()





