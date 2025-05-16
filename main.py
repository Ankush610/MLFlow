import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import logging
import os 
import mlflow
import yaml

from sklearn.model_selection import train_test_split , StratifiedKFold , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import dagshub

#===================================================================================================================

logs_dir = './logs'
os.makedirs(logs_dir,exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel('DEBUG')

fileHandler = logging.FileHandler(os.path.join(logs_dir,'model_building.log'))
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)


#===================================================================================================================

#loading data 
try:
    df = pd.read_csv('./data/preprocessed_data.csv')
    logger.info(f"Data loaded successfully..!!!")
except Exception as e:
    logger.error(f"loading data caused error : {e}")

#loading params 
try:
    with open('./params.yaml','r') as file:
        data_params = yaml.safe_load(file)
    logger.info("params loaded successfully..!!!")
except Exception as e:
    logger.error(f"loading params caused error : {e}")

#initializing mlflow 
try:
    dagshub.init(repo_owner='Ankush610', repo_name='mlflow', mlflow=True)

    mlflow.set_tracking_uri('https://dagshub.com/Ankush610/mlflow.mlflow')
    mlflow.set_experiment("RandomForest-InsurancePrediction-Exp-2")
    logger.info("mlflow x dagshub initialized sucessfully..!!!")
except Exception as e:
    logger.error(f"mlflow x dagshub initialization caused error : {e}")

#===================================================================================================================

try:
    models = [
        
        ('RandomForest',RandomForestClassifier(),{"n_estimators": [100, 200, 300],
                                                "max_depth":[None,10,20,30],
                                                "min_samples_split":[2,5,10],
                                                "min_samples_leaf":[1,2,4]}),

        ("DecisionTree",DecisionTreeClassifier(),{'criterion':['gini','entropy'],
                                                "max_depth":[None,10,20,30],    
                                                "min_samples_split":[2,5,10],
                                                "min_samples_leaf":[1,2,4]})
    ]
    logger.info("Models tuple initialized successfully..!!!")
except Exception as e:
    logger.error(f"models tuple initialization caused error : {e}")

#===================================================================================================================

for model_name , model , params in models:
    with mlflow.start_run(run_name=model_name):


        dfp = pd.read_csv('./data/original_data.csv')

        for col in dfp.columns:
            if dfp[col].dtype == 'object':

                plt.figure(figsize=(8, 5))
                sns.barplot(x=dfp[col].value_counts().index,
                            y=dfp[col].value_counts().values)
                plt.title(f"Value Counts for {col}")
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Save plot to a file
                artifact_path = f"{col}_value_counts.png"
                plt.savefig(artifact_path)
                plt.close()

                # Log the artifact to MLflow
                mlflow.log_artifact(artifact_path)
        

        try:
            x = df.drop(columns=['insuranceclaim'])
            y = df['insuranceclaim']
    
            test_size=data_params['train_test_params']['test_size']
            random_state=data_params['train_test_params']['random_state']
            shuffle=data_params['train_test_params']['shuffle']

            x_train , x_test , y_train , y_test = train_test_split(x,y,
                                                                    test_size=test_size,
                                                                    random_state=random_state,
                                                                    shuffle=shuffle)

            mlflow.log_param("test_size",test_size)
            mlflow.log_param("random_state",random_state)
            mlflow.log_param("shuffle",shuffle)

            logger.info("data splitted with test size : %s , random state : %s , shuffle : %s" % (test_size,random_state,shuffle))
            logger.info(f"train data shape : {x_train.shape} , test data shape : {x_test.shape}")

        except Exception as e:
            logger.error(f"data splitting caused error : {e}")
        

    #===================================================================================================================

        try:
            kf_n_splits=data_params['kfold_params']['n_splits']
            kf_shuffle=data_params['kfold_params']['shuffle']
            kf_random_state=data_params['kfold_params']['random_state']

            kfold = StratifiedKFold(n_splits=kf_n_splits,
                                    shuffle=kf_shuffle,
                                    random_state=kf_random_state)

            mlflow.log_param("k_folds_n_splits",kf_n_splits)
            mlflow.log_param("k_folds_random_state",kf_random_state)
            mlflow.log_param("k_folds_stratify",kf_shuffle)

            logger.info(f"StratifiedKFold initialized with n_splits : {kf_n_splits} , shuffle : {kf_shuffle} , random_state : {kf_random_state}")
            
        except Exception as e:
            logger.error(f"StratifiedKFold initialization caused error : {e}")

        try:
            gcv = GridSearchCV(estimator=model,
                                param_grid=params,
                                scoring='accuracy',
                                cv=kfold,
                                n_jobs=-1,
                                verbose=1)
            
            gcv.fit(x_train,y_train)
            logger.info(f"GridSearchCV fitted successfully..!!!")
        except Exception as e:  
            logger.error(f"GridSearchCV fitting caused error : {e}")

        try:
            best_model = gcv.best_estimator_
            best_params = gcv.best_params_
            best_score = gcv.best_score_

            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)

            mlflow.log_param("accuracy_score",best_score)

            mlflow.sklearn.log_model(best_model,model_name)

            logger.info(f"Best model : {best_model} , Best params : {best_params} , Best score : {best_score}")

        except Exception as e:  
            logger.error(f"best model extraction caused error : {e}")
    