"""Research configuration for the preserved Nilda classifiers."""
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
VALUES = {
    "TRAJECTORY_HEIGHT": 602,
    "TRAJECTORY_WIDTH": 369,
    "ROOM_X_MID": 300,
    "ROOM_Y_MID": 150,
    "RESULT_THRESHOLD": 0.5,
    "NUM_FEATURES": 4,
}
def Config(name):
    return VALUES[name]
