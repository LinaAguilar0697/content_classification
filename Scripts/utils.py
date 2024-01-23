import pandas as pd
import json
import datetime
import math
import random
import datetime
import itertools
import numpy as np

def get_embeddings(df: pd.DataFrame, column: str, embedding_name:str, model): 
    df_embedding = []

    # Get embedding from GCP Model
    for i in range(0,len(df),50):
        if i >= len(df)-50:
            start = i
            end = len(df)
        else:
            start = i
            end = i+50

        df_sample = df[column][start:end].to_list()

        embeddings = model.get_embeddings(
            df_sample
        )

        df_embedding.append(embeddings)

    # Flat embeddings list
    df_embedding_explode = list(itertools.chain(*df_embedding))
    df_embedding_explode = [element.values for element in df_embedding_explode]

    # Convert the list of lists to a NumPy array
    numpy_array = np.array(df_embedding_explode)

    # Save the NumPy array to a file, for example, as a binary .npy file
    np.save(embedding_name, numpy_array)