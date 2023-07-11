import pandas as pd
import numpy as np
import os
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


if not os.path.isdir("assets/hist_json"):
    os.mkdir("assets/hist_json")


df = pd.read_parquet("../assets/df_application_train.parquet")

for c in df.columns.to_list():
    if c=="SK_ID_CURR" or c=="TARGET": continue
    try:
        hist = np.histogram(df.loc[~df[c].isna(), c], bins=100)
        with open(f"""assets/hist_json/{c.replace("/", "_")}.json""", "w") as write_file:
            json.dump(hist, write_file, cls=NumpyArrayEncoder)
    except Exception as e:
        print(f"skipping {c}: {e}")