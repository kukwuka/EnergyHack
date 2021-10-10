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

@app.get("/vav")
def get_valves(
    QGRS_1: float, QGRS_2: float, QPlant_1: float, QPlant_2: float, QPlant_3: float, QPlant_4: float,
    PGRS_1: float, PGRS_2: float, P_1: float, P_2: float, P_3: float, P_4: float, P_5: float, P_6: float, P_7: float,
    P_8: float, P_9: float, Q_1: float, Q_2: float, Q_3: float, Q_4: float, Q_5: float, Q_6: float, Q_7: float
):
    with open('models_1.pkl', 'rb') as f:
        models = pickle.load(f)
    data = [
        QGRS_1, QGRS_2, QPlant_1, QPlant_2, QPlant_3, QPlant_4,
        PGRS_1, PGRS_2, P_1, P_2, P_3, P_4, P_5, P_6, P_7,
        P_8, P_9, Q_1, Q_2, Q_3, Q_4, Q_5, Q_6, Q_7,
        P_1 - P_2, P_2 - P_3, P_3 - P_4, P_4 - P_5, 
        P_5 - P_6, P_6 - P_7, P_7 - P_8, P_8 - P_9
    ]
    return [model.predict(data) for model in models]

@app.get("/v")
def get_optimal_valve(
    valve_1: float, valve_2: float, valve_3: float, valve_4: float, valve_5: float, valve_6: float, 
    valve_7: float, valve_8: float, valve_9: float, valve_10: float, valve_11: float, valve_12: float,
    QGRS_1: float, QGRS_2: float, PGRS_1: float, PGRS_2: float
):
    with open('models_2.pkl', 'rb') as f:
        models = pickle.load(f)

    def black_box_function(
        valve_1, valve_2, valve_3, valve_4, valve_5, valve_6, 
        valve_7, valve_8, valve_9, valve_10, valve_11, valve_12
    ):
        result = 0
        input_data = [
            valve_1, valve_2, valve_3, valve_4, valve_5, valve_6, 
            valve_7, valve_8, valve_9, valve_10, valve_11, valve_12
        ] + [QGRS_1, QGRS_2, PGRS_1, PGRS_2]
        for model in models[:6]:
            pred = model.predict(input_data)
            if pred > 0.6:
                result += model.predict(input_data)
            else:
                result -= 10

        pred_Qplant_1 = models[6].predict(input_data)
        if pred_Qplant_1 > 0.4:
            result += pred_Qplant_1
        else:
            result += - (0.4 - pred_Qplant_1)

        result += pred_Qplant_1

        pred_Qplant_2 = models[7].predict(input_data)
        if pred_Qplant_2 > 1.2:
            result += pred_Qplant_2
        else:
            result += - (1.2 - pred_Qplant_4)

        pred_Qplant_3 = models[8].predict(input_data)
        if pred_Qplant_3 > 0.4:
            result += pred_Qplant_3
        else:
            result += - (0.4 - pred_Qplant_3)

        pred_Qplant_4 = models[9].predict(input_data)
        if pred_Qplant_4 > 1.2:
            result += pred_Qplant_4
        else:
            result += - (1.2 - pred_Qplant_4)

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
        n_iter=50, # 100 is better
    )
    return optimizer.max['params']