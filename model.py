import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



df = pd.read_csv('Breast_cancer_data.csv')
mms = MinMaxScaler()
for i in df.columns:
    df[i] = mms.fit_transform(df[[i]])

plt.figure(figsize = (20,20))

ax = sns.heatmap(df.corr(), annot=True, linewidths=.5)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

inputs = df.iloc[:, :-1]
target = df.iloc[:,-1]


model_params = {
    'LinearRegression':{
        'model': LinearRegression(),
        'params':{
            'fit_intercept': [False, True],
            'normalize': [False, True],
            'copy_X': [False, True]
        }
    },
    'LogisticRegression':{
        'model': LogisticRegression(),
        'params':{
            'multi_class': ['auto', 'ovr'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'C' : [1,2,3,4,5,10],
            'max_iter' : [10, 20, 30, 50, 100,800,1000],
            'tol': [1e-3,1e-4,1e-5,1e-6]
        }
    },
    'SVC':{
        'model': SVC(),
        'params': {
            'C' : [1,2,3,4,5,10],
            'kernel' : ['rbf','linear', 'poly', 'sigmoid'],
            'degree' : [1,2,3,4,5,10],
            'gamma': ['auto', 'scale'],
            'decision_function_shape': ['ovo','ovr'],
            'tol': [1e-3,1e-4,1e-5,1e-6]
        }
    },
    'DecisionTreeClassifier':{
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10,20,50,100,150,200],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
    },
    'DecisionTreeRegressor': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random'],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [10,20,50,100,150,200],
            'criterion': ['mse', 'mae'],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'GausssianNB': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-09, 1e-10, 1e-11, 1e-12]
        }
    },
}

#RSCV Template

scores_rscv =[]
for model_name, mp in model_params.items():
    rscv_clf = RandomizedSearchCV(mp['model'], mp['params'], cv = 5, return_train_score = False)
    rscv_clf.fit(inputs, target.values.ravel())
    scores_rscv.append({
        'Model Name': model_name,
        'Best Score': rscv_clf.best_score_,
        'Best Parameter': rscv_clf.best_params_
    })
result_rscv = pd.DataFrame(scores_rscv, columns = ['Model Name', 'Best Score', 'Best Parameter'])
result_rscv




clf = RandomForestClassifier(class_weight = 'balanced_subsample', 
                             criterion =  'entropy', 
                             max_features = 'log2', 
                             n_estimators = 200)

clf.fit(inputs, target.values.ravel())

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.25)
y_pred = clf.predict(X_test)

cf_matrix = confusion_matrix(y_test, y_pred)


clf.score(X_test, y_test)
