from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pickle

from bayes_opt import BayesianOptimization
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd


app = FastAPI()

app.mount("/static", StaticFiles(directory="dist"), name="static")

@app.post("/q")
def get_q(
    valve_1: float, valve_2: float, valve_3: float, 
        valve_4: float, valve_5: float, valve_6: float, 
        valve_7: float, valve_8: float, valve_9: float, 
        valve_10: float, valve_11: float, valve_12: float
):
    valves={
        'valve_1': valve_1, 'valve_2': valve_2, 'valve_3':valve_3, 
        'valve_4': valve_4, 'valve_5': valve_5, 'valve_6':valve_6, 
        'valve_7': valve_7, 'valve_8': valve_8, 'valve_9': valve_9, 
        'valve_10': valve_10, 'valve_11': valve_11, 'valve_12': valve_12
    }
    print(valves)
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)
    return [model.predict(list(valves.values())) for model in models]

# @app.get("/q")
# def get_optimal_valve():
#     def black_box_function(
#         valve_1, valve_2, valve_3, valve_4, valve_5, valve_6, 
#         valve_7, valve_8, valve_9, valve_10, valve_11, valve_12
#     ):
#         """
#         Function with unknown internals we wish to maximize.

#         This is just serving as an example, for all intents and
#         purposes think of the internals of this function, i.e.: the process
#         which generates its output values, as unknown.
#         """
#         result = 0
#         input_data = [
#             valve_1, valve_2, valve_3, valve_4, valve_5, valve_6, 
#             valve_7, valve_8, valve_9, valve_10, valve_11, valve_12
#         ]
#         for model in models[:6]:
#             pred = model.predict(input_data)
#             if pred > 0.6:
#                 result += model.predict(input_data)
#             else:
#                 result -= 10
#         return result

#     pbounds = {
#         'valve_1': (0, 1), 'valve_2': (0, 1), 'valve_3': (0, 1), 
#         'valve_4': (0, 1), 'valve_5': (0, 1), 'valve_6': (0, 1), 
#         'valve_7': (0, 1), 'valve_8': (0, 1), 'valve_9': (0, 1), 
#         'valve_10': (0, 1), 'valve_11': (0, 1), 'valve_12': (0, 1)
#     }

#     optimizer = BayesianOptimization(
#         f=black_box_function,
#         pbounds=pbounds,
#         random_state=1,
#     )

#     optimizer.maximize(
#         # init_points=2,
#         n_iter=10, # 100 is better
#     )
#     return optimizer.max['params']
