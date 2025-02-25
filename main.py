import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler
import functions_FS as func
import csv
import time
import os
import math
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process some params.')
parser.add_argument('-c', '--count')
parser.add_argument('-pd', '--predictiondays')
parser.add_argument('-id', '--informationdays')
args = parser.parse_args()
args_count = int(args.count) - 1
prediction_days = int(args.predictiondays)
information_days = int(args.informationdays)

carpeta = './results/' + str(prediction_days) + 'PD_' + str(information_days) + 'ID/'
os.makedirs(carpeta, exist_ok=True)
os.makedirs(carpeta + 'TEMP', exist_ok=True)
os.makedirs(carpeta + 'SAL', exist_ok=True)
os.makedirs(carpeta + 'UI', exist_ok=True)
os.makedirs(carpeta + 'DACUMI', exist_ok=True)
os.makedirs(carpeta + 'U', exist_ok=True)
os.makedirs(carpeta + 'V', exist_ok=True)
os.makedirs(carpeta + 'W', exist_ok=True)
os.makedirs(carpeta + 'BV', exist_ok=True)
os.makedirs(carpeta + 'CHL', exist_ok=True)

zonas = {
		'P2':['P8','P9','SP1','SP2'],
		 'P4':['P1','P2','P5','PA','SP2','SP3','SP4'],
		 'V1':['V2','V5','V6','V7','SV2','SV3','SV4'],
		 'V4':['V2','V3','SV1','SV2'],
		 'A3':['A1','A2','A5','A6','SA1','SA2'],
		 'A8':['A0','A4','A7','A9','SA2','SA3','SA4','SA5']
		 }


write_featureBatchs = True
write_globalRank = True
write_globalResults = True

for zona in zonas:

	#########Code#############################

	aux = lambda x: math.sin(math.radians((180 / 11) * x - (180 / 11)))

	# main bucle #######################################################################################################

	zonas_aux = zonas[zona]
	dataset, l_carac_v1, l_carac_v2, df_CROCO_Dacumi = func.obtenerDataset(prediction_days, information_days, zona,
																		   zonas_aux)
	dataset = func.reetiquetarDataset(dataset, l_carac_v1, l_carac_v2, df_CROCO_Dacumi, zona, prediction_days,
									  information_days)
	dataset['Month'] = dataset['Month'].apply(aux)

	df_temp, df_sal, df_ui, df_dacumi, df_u, df_v, df_w, df_bv, df_chl, def_month, df_target = func.split_datasets(dataset, zona)
	dataframes = {'TEMP':df_temp, 'SAL':df_sal, 'UI':df_ui, 'DACUMI':df_dacumi, 'U':df_u, 'V':df_v, 'W':df_w, 'BV':df_bv, 'CHL':df_chl}
	list_cars = []
	for df_tipe in dataframes:
		df = dataframes[df_tipe]
		regr = RandomForestRegressor(n_estimators=1000, random_state=0)
		scaler = MinMaxScaler()
		scaler.fit(df.iloc[:-365,:])
		X = scaler.transform(df.iloc[:-365,:])
		regr.fit(X, df_target.iloc[:-365])
		rank = {'feature': [], 'score': []}
		scores = list(regr.feature_importances_)
		scores.sort(reverse=True)
		rank['score'] = scores
		for i in range(len(df.keys())):
			rank['feature'].append(df.keys()[np.where(ss.rankdata(regr.feature_importances_ * -1) == i+1)].values[0])

		if write_featureBatchs:
			with open(carpeta + df_tipe + '/' + zona + '_' + df_tipe + '_rank.csv', 'w', newline='') as f:
				write = csv.writer(f)
				write.writerow(rank['feature'])
				write.writerow(rank['score'])
				f.close()

		for i in rank['feature'][0:5]:
			list_cars.append(i)
	list_cars.append('Month')

	df_final = dataset[list_cars]
	regr = RandomForestRegressor(n_estimators=1000, random_state=0)
	scaler = MinMaxScaler()
	scaler.fit(df_final.iloc[:-365, :])
	X = scaler.transform(df_final.iloc[:-365, :])
	regr.fit(X, df_target.iloc[:-365])
	rank = {'feature': [], 'score': []}
	print(regr.feature_importances_)
	scores = list(regr.feature_importances_)
	scores.sort(reverse=True)
	rank['score'] = scores
	for i in range(len(list_cars)):
		rank['feature'].append(list_cars[np.where(ss.rankdata(regr.feature_importances_ * -1) == i+1)[0][0]])

	if write_globalRank:
		with open(carpeta + zona + '_rank.csv', 'w', newline='') as f:
			write = csv.writer(f)
			write.writerow(rank['feature'])
			write.writerow(rank['score'])
			f.close()

	for n_features in [5, 10, 15, 20, 30, 35, 40, 45]:
		df_final2 = dataset[rank['feature'][0:n_features] + ['Dacuminata' + zona + '_output']]
		for model in ['mlp_bl', 'rf_bl', 'svr_bl']:
			inicio = time.time()
			score, params, output = func.grid_search(df_final2, zona, model, args_count)
			fin = time.time()
			if score:
				if write_globalResults:
					row = [n_features] + [model] + params + score + [fin - inicio] + output
					with open(carpeta + zona + '_results.csv', 'a+', newline='') as f:
						write = csv.writer(f)
						write.writerow(row)
						f.close()

	dataset_aux = dataset.copy()
	y = dataset_aux.pop('Dacuminata' + zona + '_output')
	y_test = y.iloc[-365:]
	with open(carpeta + zona + '_results_base.csv', 'w+', newline='') as f:
		write = csv.writer(f)
		write.writerow(y_test.tolist())
		f.close()
