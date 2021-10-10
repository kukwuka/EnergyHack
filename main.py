from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pickle

from bayes_opt import BayesianOptimization
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd




app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="dist"), name="static")

@app.get("/q")
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

@app.get("/v")
def get_optimal_valve(
    valve_1: float, valve_2: float, valve_3: float, valve_4: float, valve_5: float, valve_6: float, 
    valve_7: float, valve_8: float, valve_9: float, valve_10: float, valve_11: float, valve_12: float
):
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)

    def black_box_function(
        valve_1, valve_2, valve_3, valve_4, valve_5, valve_6, 
        valve_7, valve_8, valve_9, valve_10, valve_11, valve_12
    ):
        """
        Function with unknown internals we wish to maximize.

        This is just serving as an example, for all intents and
        purposes think of the internals of this function, i.e.: the process
        which generates its output values, as unknown.
        """
        result = 0
        input_data = [
            valve_1, valve_2, valve_3, valve_4, valve_5, valve_6, 
            valve_7, valve_8, valve_9, valve_10, valve_11, valve_12
        ]
        for model in models[:6]:
            pred = model.predict(input_data)
            if pred > 0.6:
                result += model.predict(input_data)
            else:
                result -= 10

        pred_Qplant_1 = models[6].predict(input_data)
        if pred_Qplant_1 > 0.4:
            result += pred_Qplant_1.predict(input_data)
        else:
            result += 0.4 - pred_Qplant_1.predict(input_data)
         
        result += pred_Qplant_1.predict(input_data)

        pred_Qplant_2 = models[7].predict(input_data)
        if pred_Qplant_2 > 1.4:
            result += pred_Qplant_2.predict(input_data)
        else:
            result += 1.4 - pred_Qplant_2.predict(input_data)

        pred_Qplant_3 = models[8].predict(input_data)
        result += pred_Qplant_3.predict(input_data)

        pred_Qplant_4 = models[9].predict(input_data)
        if pred_Qplant_4 > 1.6:
            result += pred_Qplant_4.predict(input_data)
        else:
            result += 1.6 - pred_Qplant_4.predict(input_data)

        return result

    pbounds = {
        'valve_1': (0, 1) if valve_1 == -1 else (valve_1, valve_1), 
        'valve_2': (0, 1) if valve_2 == -1 else (valve_2, valve_2), 
        'valve_3': (0, 1) if valve_3 == -1 else (valve_3, valve_3), 
        'valve_4': (0, 1) if valve_4 == -1 else (valve_4, valve_4), 
        'valve_5': (0, 1) if valve_5 == -1 else (valve_5, valve_5), 
        'valve_6': (0, 1) if valve_6 == -1 else (valve_6, valve_6), 
        'valve_7': (0, 1) if valve_7 == -1 else (valve_7, valve_7), 
        'valve_8': (0, 1) if valve_8 == -1 else (valve_8, valve_8), 
        'valve_9': (0, 1) if valve_9 == -1 else (valve_9, valve_9), 
        'valve_10': (0, 1) if valve_10 == -1 else (valve_10, valve_10), 
        'valve_11': (0, 1) if valve_11 == -1 else (valve_11, valve_11), 
        'valve_12': (0, 1) if valve_12 == -1 else (valve_12, valve_12)
    }

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        # init_points=2,
        n_iter=10, # 100 is better
    )
    return optimizer.max['params']
