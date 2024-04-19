import os
import sys
from dataclasses import dataclass

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
import numpy as np

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_trainer(self,train_arr,test_arr):
        try:

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            svc = SVC(kernel='linear')

            svc.fit(X_train, y_train)

            ypred = svc.predict(X_test)


            test_acc = f"Test accuracy score of model is {round(accuracy_score(y_test, ypred),4)*100}%"


            logging.info(f"{test_acc}")

            logging.info(f"Saved model object.")

            save_object(

                file_path=self.model_trainer_config.model_file_path,
                obj=svc

            )


            return (
                test_acc
            )

        except Exception as e:
            raise CustomException(e, sys)