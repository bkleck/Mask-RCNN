import pandas as pd
import sklearn
from sklearn.neighbors import BallTree, KDTree
import numpy as np
import os
import logging
import argparse
import shutil
from shutil import copy

# create logging configs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/')
args = parser.parse_args()

# get the path to the data folder
# user has to input the directory (e.g. Field Test No.1)
logging.info(f'-- Started extraction of excels---')
input_dir = os.path.join('V:/', args.data_dir)
results_path = os.path.join(input_dir, 'Results')
server_results = os.listdir(results_path)

# function to extract prediction excel sheets to the data folder
def extract_predictions(input_folder, results_path, output_path):
    input_path = os.path.join(results_path, input_folder)
    preds = os.path.join(input_path, 'Predictions')

    # get the list of predictions
    predictions = os.listdir(preds)

    for folder in predictions:
        pth = os.path.join(preds, folder)
        data_path = os.path.join(pth, 'Data')
        excels = os.listdir(data_path)
        excel = os.path.join(data_path, excels[0])
        # copy the file to the data folder
        copy(excel, output_path)
        logging.info(f'Completed copying excel for {excels[0]}.')

# get current path and add on /data to that path
curr_path = os.getcwd()
data_path = os.path.join(curr_path, 'data')

for server in server_results:
    extract_predictions(server, results_path, data_path)
logging.info(f'Completed extraction of excels!\n')

# defect_df = pd.read_excel('data/GH020052_prediction_data.xlsx', index_col=0)
# lamp_df = pd.read_csv('lamps.csv')

# # function to find nearest lamp to the detected defect
# # output the original defect dataframe together with appended lamp data of nearest lamp
# def find_nearest(file, lamp_df, data_path):

#     # open the excel in a pandas dataframe
#     file_path = os.path.join(data_path, file)
#     defect_df = pd.read_excel(file_path)

#     # Creates new columns converting coordinate degrees to radians for both dataframes
#     for column in defect_df[["Latitude", "Longitude"]]:
#         rad = np.deg2rad(defect_df[column].values)
#         defect_df[f'{column}_rad'] = rad

#     for column in lamp_df[["Latitude", "Longitude"]]:
#         rad = np.deg2rad(lamp_df[column].values)
#         lamp_df[f'{column}_rad'] = rad
    
#     # Takes the first group's latitude and longitude values to construct the ball tree algo
#     logging.info(f'Started calculating nearest lamp for {file}.')
#     ball = BallTree(lamp_df[["Latitude_rad", "Longitude_rad"]].values, metric='haversine')

#     # The amount of neighbors to return.
#     k = 1

#     # Executes a query with the second group. This will also return two arrays
#     distances, indices = ball.query(defect_df[["Latitude_rad", "Longitude_rad"]].values, k = k)
#     # logging.info(f'Completed calculation for {defect_df}.')

#     # create new lists to store the name and ID of the nearest lamps
#     nearest_lamps = []
#     nearest_id = []

#     # for each index returned from the balltree algo, 
#     # we will use that index to extract the corresponding nearest lamp to match to the defect
#     for index in indices:
#         index = index[0]
#         lamp = lamp_df.iloc[index, [3]][0]
#         id = lamp_df.iloc[index, [4]][0]
#         nearest_lamps.append(lamp)
#         nearest_id.append(id)

#     defect_df['lamp'] = nearest_lamps
#     defect_df['lamp-id'] = nearest_id
#     return defect_df

# # list files within a directory that end with .xlsx
# files = os.listdir(data_path)
# excel_files = [file for file in files if file.endswith('xlsx')]

# # iterate through each excel file and run the nearest lamp function
# for file in excel_files:
#     df = find_nearest(file, lamp_df, data_path)
#     file_path = os.path.join(data_path, file)
#     df.drop(columns=['Latitude_rad', 'Longitude_rad'], inplace=True)
#     df.to_excel(file_path, index=False)
#     logging.info(f'Successfully completed for {file}!\n')
