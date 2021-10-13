import json
from channels.generic.websocket import WebsocketConsumer
from .Utils.general import is_valid_input_payload, str_bytes_to_pandas_df
from types import SimpleNamespace

from .MachineLearning.MachineLearning import HyperparameterTuning

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

        algorithm_name = payload_dict["instructions"].get("algorithm_name_str")
        search_name = payload_dict["instructions"].get("search_name_str")
        param_grid = payload_dict["instructions"].get("param_grid")

        tuning_model = HyperparameterTuning(X= Data.X,
                                            y= Data.y,
                                            algorithm_str= algorithm_name,
                                            search_str= search_name,
                                            hyperparams_grid= param_grid)

        res_dict = tuning_model.run_search()
        print(f'Res dict is {res_dict}')

        # Sends back the best model we have obtained through the procedure
        self.send(json.dumps(
            res_dict
        ))