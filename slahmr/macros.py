import os
from pathlib import Path

BASE_OUTPUT_PATH = os.path.join(str(Path(__file__).resolve().parent.parent), "outputs")
DATASET_BASE_PATH = os.environ.get("EGOEXO4D_DATASET")
if DATASET_BASE_PATH is None:
    raise ValueError("EGOEXO4D_DATASET is not set. Make sure to download the dataset and set the EGOEXO4D_DATASET environment variable.")

if not os.path.exists(BASE_OUTPUT_PATH):
    print(f"Creating base output path: {BASE_OUTPUT_PATH}")
    os.makedirs(BASE_OUTPUT_PATH)

CAM_NAME2ID = {
    "cam01": 1,
    "cam02": 2,
    "cam03": 3,
    "cam04": 4,
    "aria": 5,
}