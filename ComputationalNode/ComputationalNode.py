import json
from channels.generic.websocket import WebsocketConsumer
from ast import literal_eval
from .Utils.general import is_valid_input_payload, str_bytes_to_pandas_df
from .Utils.algorithm_switcher import algorithm_switch
import pandas as pd
from types import SimpleNamespace
from sklearn.model_selection import GridSearchCV #change to bayesian
from django.utils import timezone
import datetime


class ComputationalNode(WebsocketConsumer):

    def connect(self):

        # 1. Verify where the request is coming from, and save the node as orchestrator
        # Note that you can use the self.scope attribute to retrieve important data
        orchestrator_ip = self.scope['client'][0]
        print("Accepting incoming connection from IP", orchestrator_ip)

        # 2. Check if the orchestrator IP is available in the verified DB of orchestrator IPs
        #pseudocode -> if orchestratorIP in db.query(verified_orchestrator_IPs) then conferm else refuse

        # 3. Accept the connection
        self.accept()

        # 4. Send back notification of acceptance of the connection
        self.send(json.dumps({
            "status": 200
        }))


    def receive(self, text_data=None, bytes_data=None):
        # Analyze the payload here, check that all of them comply with the predefined format
        decoded_bytes_data = bytes_data.decode('utf-8')

        #Change the structure of the below
        payload_dict = is_valid_input_payload(decoded_bytes_data)
        print('Data Received just now.')

        # Decodes all the inputs and gets them back to being pandas dataframes
        data_dict = {k: str_bytes_to_pandas_df(v) for k, v in payload_dict.get('data').items()}

        # Saves the dict in the object Data
        Data = SimpleNamespace(**data_dict)

        algorithm_name = payload_dict["instructions"].get("algorithm_name")
        param_grid = payload_dict["instructions"].get("param_grid")

        # Creates an instance of the classifier, if you want to insert random states and so forth do it here
        classifier = algorithm_switch(algorithm_name)()

        # Creates the GridSearch object which takes as an input the specific classifier with all CPU
        grid_search_cv = GridSearchCV(estimator=classifier,
                                      param_grid=param_grid,
                                      n_jobs=-1)

        # Starts the cross validation procedure
        fitted_grid_search_cv = grid_search_cv.fit(Data.X, Data.y)

        # Sends back the best model we have obtained through the procedure
        self.send(json.dumps({
            'best_params': fitted_grid_search_cv.best_params_,
            'best_score': fitted_grid_search_cv.best_score_,
        }))

        print(f'Best params are: {fitted_grid_search_cv.best_params_}')
        print(f'Best score is: {fitted_grid_search_cv.best_score_} using scorer {fitted_grid_search_cv.scorer_}')