import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import geopandas as gpd
import reverse_geocoder as rg
import pycountry
import pymysql
from shapely.geometry import MultiPoint
from collections import Counter
from IPython.display import display
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sqlalchemy import create_engine
import logging

from image_processing import server_predict
import Server_Module # Python Module



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def main():

    folder_path = "D:\\ubuntu\\RIDA\\Satellite_Image"
    output_folder_after ="D:\\RIDA\\Sentinel_Process\\Image"

    server_predict(folder_path, output_folder_after)

    # Define the directory containing CSV files
    data_dir = r"D:\Work\Code งาน\Lab-docker\RIDA\RIDA_CSV\Predict\All_Predict"

    # Loop through all files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
        # Construct the full path to the CSV file
            data_path = os.path.join(data_dir, filename)

            # Perform Prerpocessing
            df = pd.read_csv(data_path) # Read CSV in specify directory
            scaler_path = (r'D:\Work\Code งาน\Lab-docker\RIDA\Export Model\04-06-2024\min_max_scaler.pkl') # Directory to Mormalization Pickle File
            df_rename, lat, long, fire_date = Server_Module.preprocess(df, scaler_path) # Call Preprocessing Function

            # Load Model


if __name__ == "__main__":
    main()