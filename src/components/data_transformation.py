import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.exceptions import CustomException
from src.logger import logging
import os

from src.utils import save_object


class DataTransformation:
    def __init__(self):
        preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")


    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            target_column_name = "prognosis"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # logging.info(
            #     f"Applying preprocessing object on training dataframe and testing dataframe."
            # )
            #
            # input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            logging.info(f"Train and test array")

            return (
                train_arr,
                test_arr,
            )
        except Exception as e:
            raise CustomException(e, sys)