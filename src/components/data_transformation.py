import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preproccesor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class Datatransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            num_colums=["writing_score","reading_score"]
            cat_columns=[
                "gender","parental_level_of_education","lunch","race_ethnicity","test_preparation_course",
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("numerical columns scaling completed")
            cat_pipline=Pipeline(
                [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("categorical columns encoding completed")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_colums),
                    ("cat_pipline",cat_pipline,cat_columns)
                ]
            )


            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)



    def initiate_data_transform(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test is completed")
            logging.info("obtain preprocessor object")
            preprocessor_obj=self.get_data_transformer_object()

            target_columns_name="math_score"
            num_colums=["writing_score","reading_score"]

            input_feat_train_df=train_df.drop(columns=[target_columns_name],axis=1)
            target_feat_train_df=train_df[target_columns_name]

            input_feat_test_df=test_df.drop(columns=[target_columns_name],axis=1)
            target_feat_test_df=test_df[target_columns_name]

            logging.info("applying object on training and testing")

            input_feat_train_arr=preprocessor_obj.fit_transform(input_feat_train_df)
            input_feat_test_arr=preprocessor_obj.fit_transform(input_feat_test_df)

            train_arr=np.c_[
                input_feat_train_arr,np.array(target_feat_train_df)
            ]
            test_arr=np.c_[input_feat_test_arr,np.array(target_feat_test_df)]
            save_obj(
                file_path=self.data_transformation_config.preproccesor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preproccesor_obj_file_path
            )
        except Exception as e:
           raise CustomException(e,sys)