import websocket, json, random, time
import hashlib
import pandas as pd
import os
import pickle, codecs
import asyncio
import websockets
import nest_asyncio
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm

df = sns.load_dataset('iris')
X = df.drop('species', axis=1).values
y = df['species'].values

X_pickled = codecs.encode(pickle.dumps(X), "base64").decode()
y_pickled = codecs.encode(pickle.dumps(y), "base64").decode()

payload = {
    "metadata": {
        "X": X_pickled,
        "y": y_pickled,
    },
    "instructions": {
        "param_grid" : {'n_estimators': [5, 10, 50, 100], 'criterion': ['giny', 'entropy']},
        "algorithm_name": 'random_forest',
        "search_name_str": 'grid_search_cv'
    },
}

nest_asyncio.apply()


encoded_payload = json.dumps(payload).encode('utf-8')

async def send_data():
    async with websockets.connect('ws://192.168.1.227:8000/compute/') as websocket:
        pp = json.dumps(payload).encode('utf-8')
        await websocket.send(pp)
        response = await websocket.recv()
        print(response)

r = asyncio.get_event_loop().run_until_complete(send_data())
