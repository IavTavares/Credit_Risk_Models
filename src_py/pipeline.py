
import constants

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import optuna
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, KFold, cross_val_score
from sklearn.metrics import classification_report, RocCurveDisplay, roc_curve,\
roc_auc_score,precision_recall_fscore_support
from mlflow.exceptions import MlflowException


def preprocessing():
    df_raw = pd.read_csv(constants.DATA_FOLDER_PATH+"/credit_risk_dataset.csv")
    # home_ownership
    pers_h_own_label=LabelEncoder()
    pers_h_own_label.fit(df_raw["person_home_ownership"])
    df_raw["person_home_ownership_enc"]=pers_h_own_label.transform(df_raw["person_home_ownership"]);
    # loan_intent
    loan_intent_label=LabelEncoder()
    loan_intent_label.fit(df_raw["loan_intent"])
    df_raw["loan_intent_enc"]=loan_intent_label.transform(df_raw["loan_intent"]);
    # loan_grade
    loan_grade_label=LabelEncoder()
    loan_grade_label.fit(df_raw["loan_grade"])
    df_raw["loan_grade_enc"]=loan_grade_label.transform(df_raw["loan_grade"]);
    # NaNs
    df_raw=df_raw.dropna()
    df_raw.to_csv(constants.DATA_FOLDER_PATH+"/Data_No_NaN.csv",index=False)

    Y=df_raw["loan_status"]
    X=df_raw[["person_age","person_income","person_home_ownership_enc",
            "person_emp_length",'loan_intent_enc',
            'loan_grade_enc','loan_amnt','loan_int_rate','loan_percent_income']]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,
                    random_state=123 # just for replicability. remove later. 
                    )
    # 
    return [X_train, X_test, Y_train, Y_test]

def lr_mlflow(experiment,data:list):
    """experiment is a mlflow experiment object"""
    [X_train, X_test, Y_train, Y_test] = data
    reg_base=LogisticRegression()
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Logistic_Regression"):
        reg_base.fit(X_train, np.ravel(Y_train))
        #mlflow.log_params([])
        Y_pred=reg_base.predict(X_test)
        preds= reg_base.predict_proba(X_test)
        preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])
        preds_df["loan_status"] = preds_df["prob_default"].apply(lambda x: 1 if x > 0.5 else 0)
        target_names = ['Non-Default', 'Default']
        
        # mlflow.log_artifact(classification_report(Y_test,preds_df["loan_status"], target_names=target_names))
        
        prob_default =preds_df["prob_default"] # preds[:, 1]
        fallout, sensitivity, thresholds = roc_curve(Y_test, prob_default)
        accuracy = reg_base.score(X_test, Y_test)
        mlflow.log_metric("accuracy", accuracy)
        # mlflow.log_metric("fallout", fallout)
        # mlflow.log_metric("sensitivity", sensitivity)
        # mlflow.log_metric("thresholds", thresholds)
        # number of loan defaults from the prediction data
        pd.DataFrame({'fallout':fallout, 
                    'sensitivity':sensitivity}).to_csv(
                    os.path.join(constants.MLFLOW_ARTIFACT_PATH,"LogReg",
                    "Roc_Logistic.csv"),index=False
                    )
        
        n_defaults = preds_df["loan_status"].value_counts()[1]
        mlflow.log_metric("predicted number of defaults", n_defaults)
        # default recall - True Default (Positive) Rate : proportion of correctly identified defaults
        # out of all data defaults
        default_recall = precision_recall_fscore_support(Y_test,preds_df["loan_status"])[1][1]
        mlflow.log_metric("prop of correctly identified defaults", default_recall)
        # Calculate the estimated impact of the new default recall rate
        avg_loan_amnt = X_test["loan_amnt"].mean()
        default_rr=n_defaults * avg_loan_amnt * (1 - default_recall)
        mlflow.log_metric("estimated prop of avg loss misid defaults", default_rr)
        # creating a plot
        plt.plot(fallout, sensitivity, color = 'darkorange', 
            label = "Logistic Regression + Threshold")
        plt.plot([0, 1], [0, 1], linestyle='--', label="Random Classifier")
        plt.legend(title='Receiver Operating Characteristic Curves')
        plt.savefig(
            os.path.join(
            constants.MLFLOW_ARTIFACT_PATH,"LogReg","ROCcurve.png"
            )
        )
        # plt.show()
        # plt.close()
        mlflow.log_artifacts(
            local_dir = os.path.join(
                constants.MLFLOW_ARTIFACT_PATH,"LogReg"
            )
        )

if __name__ == "__main__":
    data = preprocessing()
    try:
        experiment= mlflow.create_experiment("Credit Risk Experiments",
                                            artifact_location = "mlflow_artifacts")
    except MlflowException:
        experiment = mlflow.get_experiment_by_name("Credit Risk Experiments")

    lr_mlflow(experiment,data)