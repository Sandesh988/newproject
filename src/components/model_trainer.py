import os 
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj,evaluate

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

@dataclass
class ModelTrainconfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')

class modeltrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainconfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split training and testing input data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            models={
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "gradiantboost":GradientBoostingRegressor(),
            }
            model_report:dict=evaluate(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)


            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info("best found model on both training and test data")
            
            
            save_obj(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_sco=r2_score(y_test,predicted)
            return r2_sco

        except Exception as e:
            raise CustomException(e,sys)


       

