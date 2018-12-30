from dotenv import load_dotenv
from os import environ
from typing import Dict, Any

load_dotenv()

_SETTINGS = {
    'CLASSIFIER': 'keras',
    'ONE_HOT_ENCODE': True,
    'NORMALIZE': True,
    'GRID_PARAMS': dict(
        keras__hidden_layers=[1, 2, 3],
        keras__hidden_units=[1, 6, 12, 24],
        keras__dropout_rate=[0.05, 0.1, 0.3],
        keras__optimizer=['adam', 'sgd']
    )
}


def settings() -> Dict[str, Any]:
    """
    Returns a unified dict of all settings defined above as well as any ENV variables
    :return:
    """
    return {**_SETTINGS, **environ}
