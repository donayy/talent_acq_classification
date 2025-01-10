import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, plot_roc_curve
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import cross_val_score
import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore")
filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


attributes_df = pd.read_excel("datasets/attributes.xls")
potential_labels_df = pd.read_excel("datasets/potential_labels.xls")
attributes_df.head()
potential_labels_df.head()


df = attributes_df.merge(potential_labels_df, on=["task_response_id", "match_id", "evaluator_id", "player_id"])

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df = df[df.position_id != 1]

df = df[df.potential_label != "below_average"]

df_new = df.pivot_table(index=["player_id","position_id", "potential_label"], columns=["attribute_id"], values="attribute_value")

df_new = df_new.rename_axis(None, axis=1).reset_index()

le = LabelEncoder()
df_new["potential_label"] = le.fit_transform(df_new["potential_label"])


num_cols = [col for col in df_new.columns if col not in ["player_id", "position_id", "potential_label"]]

scaler = StandardScaler()
df_new[num_cols] = scaler.fit_transform(df_new[num_cols])
df_new[num_cols].head()


y = df_new["potential_label"]
X = df_new.drop(["player_id", "position_id", "potential_label"], axis=1)
X.columns = X.columns.astype(str)

classifiers = [('KNN', KNeighborsClassifier()),
               ("CART", DecisionTreeClassifier()),
               ("RF", RandomForestClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
               ("LightGBM", LGBMClassifier())]

def base_models_all_scores(X, y, scoring="roc_auc"):
    print("Base Models....")
    models = [('KNN', KNeighborsClassifier()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ("LightGBM", LGBMClassifier())]

    for name, model in models:
        print(f"########## {name} ##########")
        for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            cv_results = cross_val_score(model, X, y, scoring=score, cv=5).mean()
            print(f"{score} score: {round(cv_results.mean(), 4)}")

base_models_all_scores(X, y)
# Base Models....
# ########## KNN ##########
# roc_auc score: 0.8051
# f1 score: 0.4657
# precision score: 0.9333
# recall score: 0.3197
# accuracy score: 0.8523
# ########## CART ##########
# roc_auc score: 0.7567
# f1 score: 0.5895
# precision score: 0.5857
# recall score: 0.6758
# accuracy score: 0.8268
# ########## RF ##########
# roc_auc score: 0.8943
# f1 score: 0.5602
# precision score: 0.8711
# recall score: 0.5152
# accuracy score: 0.8819
# ########## GBM ##########
# roc_auc score: 0.8678
# f1 score: 0.6222
# precision score: 0.754
# recall score: 0.5485
# accuracy score: 0.8671
# ########## XGBoost ##########
# roc_auc score: 0.8714
# f1 score: 0.6259
# precision score: 0.7054
# recall score: 0.5864
# accuracy score: 0.8562
# ########## LightGBM ##########
# roc_auc score: 0.8868
# f1 score: 0.6479
# precision score: 0.7984
# recall score: 0.5682
# accuracy score: 0.8746


knn_params = {"n_neighbors": range(2, 50)}
cart_params = {'max_depth': range(1, 20), "min_samples_split": range(2, 30)}
rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 300]}
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 200]}
lgbm_params = {"learning_rate": [0.01, 0.1], "n_estimators": [300, 500], "colsample_bytree": [0.5, 0.7, 1]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ("GBM", GradientBoostingClassifier(), gbm_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            cv_results = cross_val_score(classifier, X, y, scoring=score, cv=5).mean()
            print(f"{score} score before optimization : {round(cv_results.mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=score)
            print(f"* {score} score after optimization : {cv_results['test_score'].mean()}")
        best_models[name] = final_model
    return best_models


best_models = hyperparameter_optimization(X, y);

# Hyperparameter Optimization....
# ########## KNN ##########
# roc_auc score before optimization : 0.8051
# f1 score before optimization : 0.4657
# precision score before optimization : 0.9333
# recall score before optimization : 0.3197
# accuracy score before optimization : 0.8523
# * roc_auc score after optimization : 0.7719220074617077
# * f1 score after optimization : 0.47837606837606844
# * precision score after optimization : 0.9583333333333334
# * recall score after optimization : 0.3226120857699805
# * accuracy score after optimization : 0.8560846560846561
# ########## CART ##########
# roc_auc score before optimization : 0.7668
# f1 score before optimization : 0.6125
# precision score before optimization : 0.5244
# recall score before optimization : 0.6788
# accuracy score before optimization : 0.8084
# * roc_auc score after optimization : 0.7235623781676414
# * f1 score after optimization : 0.5913217138707335
# * precision score after optimization : 0.9
# * recall score after optimization : 0.46101364522417154
# * accuracy score after optimization : 0.8781033781033781
# ########## RF ##########
# roc_auc score before optimization : 0.9058
# f1 score before optimization : 0.5861
# precision score before optimization : 0.8984
# recall score before optimization : 0.4788
# accuracy score before optimization : 0.8746
# * roc_auc score after optimization : 0.8983454822502265
# * f1 score after optimization : 0.5871338524286244
# * precision score after optimization : 0.8611111111111112
# * recall score after optimization : 0.46296296296296297
# * accuracy score after optimization : 0.8744810744810745
# ########## GBM ##########
# roc_auc score before optimization : 0.8644
# f1 score before optimization : 0.5675
# precision score before optimization : 0.7286
# recall score before optimization : 0.5303
# accuracy score before optimization : 0.8525
# * roc_auc score after optimization : 0.8690822648692675
# * f1 score after optimization : 0.6564564564564564
# * precision score after optimization : 0.7589285714285715
# * recall score after optimization : 0.5526315789473684
# * accuracy score after optimization : 0.8597476597476598
# ########## XGBoost ##########
# roc_auc score before optimization : 0.8714
# f1 score before optimization : 0.6259
# precision score before optimization : 0.7054
# recall score before optimization : 0.5864
# accuracy score before optimization : 0.8562
# * roc_auc score after optimization : 0.8581604419673772
# * f1 score after optimization : 0.6166666666666667
# * precision score after optimization : 0.7316017316017316
# * recall score after optimization : 0.5516569200779727
# * accuracy score after optimization : 0.8598290598290598
# ########## LightGBM ##########
# roc_auc score before optimization : 0.8868
# f1 score before optimization : 0.6479
# precision score before optimization : 0.7984
# recall score before optimization : 0.5682
# accuracy score before optimization : 0.8746
# * roc_auc score after optimization : 0.8456624248414464
# * f1 score after optimization : 0.6238095238095238
# * precision score after optimization : 0.8125
# * recall score after optimization : 0.5175438596491228
# * accuracy score after optimization : 0.870899470899471

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)
# Voting Classifier...
# Accuracy: 0.8634513634513635
# F1Score: 0.540952380952381
# ROC_AUC: 0.8914996095263371



def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(best_models, X)
