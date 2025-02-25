import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

sys.setrecursionlimit(9999)

def obtenerDataset(dias_prediccion, dias_caracteristicas, zona, zonas_aux):
	lista_caracteristicas_v1 = []
	lista_caracteristicas_v2 = []

	archivo = './data/DailyUI_ncep.csv'

	df_DailyUI=pd.read_csv(archivo,sep=',')
	ui = ['UI']

	archivo ='./data/STA_CROCO_DacumiReal2013a2019_gapfree.csv'

	df_CROCO_Dacumi=pd.read_csv(archivo,sep=',')
	month = ['Month']

	v2 =['TSur_mean'+zona, 'TSur_std'+zona, 'TBot_mean'+zona, 'TBot_std'+zona, 'SSur_mean'+zona, 'SSur_std'+zona, 'SBot_mean'+zona, 'SBot_std'+zona, 'USur_mean'+zona, 'USur_std'+zona, 'UBot_mean'+zona, 'UBot_std'+zona, 'VSur_mean'+zona, 'VSur_std'+zona, 'VBot_mean'+zona, 'VBot_std'+zona, 'BVmax'+zona, 'dept_BVmax'+zona, 'Chl'+zona,'Dacuminata'+zona]
	lista_caracteristicas_v1.append(v2)

	archivo ='./data/SECT_CROCO2.csv'
	df_CROCO=pd.read_csv(archivo,sep=',')

	for zona_aux in zonas_aux:
		if zona_aux[0] != 'S':
			datos_zona = ['TSur_mean' + zona_aux, 'TSur_std' + zona_aux, 'TBot_mean' + zona_aux, 'TBot_std' + zona_aux,
						  'SSur_mean' + zona_aux, 'SSur_std' + zona_aux, 'SBot_mean' + zona_aux, 'SBot_std' + zona_aux,
						  'USur_mean' + zona_aux, 'USur_std' + zona_aux, 'UBot_mean' + zona_aux, 'UBot_std' + zona_aux,
						  'VSur_mean' + zona_aux, 'VSur_std' + zona_aux, 'VBot_mean' + zona_aux, 'VBot_std' + zona_aux,
						  'BVmax' + zona_aux, 'dept_BVmax' + zona_aux, 'Chl' + zona_aux, 'Dacuminata' + zona_aux]
			lista_caracteristicas_v1.append(datos_zona)
		else:
			datos_zona = ['TSur1_mean' + zona_aux, 'TSur1_std' + zona_aux, 'TBot1_mean' + zona_aux,
						  'TBot1_std' + zona_aux,
						  'SSur1_mean' + zona_aux, 'SSur1_std' + zona_aux, 'SBot1_mean' + zona_aux,
						  'SBot1_std' + zona_aux,
						  'USur1_mean' + zona_aux, 'USur1_std' + zona_aux, 'UBot1_mean' + zona_aux,
						  'UBot1_std' + zona_aux,
						  'VSur1_mean' + zona_aux, 'VSur1_std' + zona_aux, 'VBot1_mean' + zona_aux,
						  'VBot1_std' + zona_aux,
						  'WSur1_mean' + zona_aux, 'WSur1_std' + zona_aux, 'WBot1_mean' + zona_aux,
						  'WBot1_std' + zona_aux,
						  'TSur2_mean' + zona_aux, 'TSur2_std' + zona_aux, 'TBot2_mean' + zona_aux,
						  'TBot2_std' + zona_aux,
						  'SSur2_mean' + zona_aux, 'SSur2_std' + zona_aux, 'SBot2_mean' + zona_aux,
						  'SBot2_std' + zona_aux,
						  'USur2_mean' + zona_aux, 'USur2_std' + zona_aux, 'UBot2_mean' + zona_aux,
						  'UBot2_std' + zona_aux,
						  'VSur2_mean' + zona_aux, 'VSur2_std' + zona_aux, 'VBot2_mean' + zona_aux,
						  'VBot2_std' + zona_aux,
						  'WSur2_mean' + zona_aux, 'WSur2_std' + zona_aux, 'WBot2_mean' + zona_aux,
						  'WBot2_std' + zona_aux
						  ]
			lista_caracteristicas_v2.append(datos_zona)

	dataset=pd.DataFrame(df_CROCO_Dacumi[month].iloc[0:df_CROCO_Dacumi.shape[0], :])
	dataset=dataset.shift(periods=((dias_caracteristicas-1) * -1))

	for i in range(dias_caracteristicas):
		dataset = pd.concat(
			[dataset, df_DailyUI[ui].iloc[0+i:df_CROCO_Dacumi.shape[0] - (dias_prediccion-i), :].reset_index(drop=True)],
			axis=1)

	for bloques_caracteristicas in lista_caracteristicas_v1:
		for i in range(dias_caracteristicas):
			dataset=pd.concat([dataset, df_CROCO_Dacumi[bloques_caracteristicas].iloc[0+i:df_CROCO_Dacumi.shape[0]-(dias_prediccion-i),:].reset_index(drop=True)], axis=1)
	for bloques_caracteristicas in lista_caracteristicas_v2:
		for i in range(dias_caracteristicas):
			dataset=pd.concat([dataset, df_CROCO[bloques_caracteristicas].iloc[0+i:df_CROCO_Dacumi.shape[0]-(dias_prediccion-i),:].reset_index(drop=True)], axis=1)
	#dataset=df_CROCO_Dacumi[v2].iloc[0:df_CROCO_Dacumi.shape[0]-dias_prediccion,:]
	#dataset['Dacuminata'+zona] = dataset['Dacuminata'+zona].where(dataset['Dacuminata'+zona] < 500, 500)

	return dataset, lista_caracteristicas_v1, lista_caracteristicas_v2, df_CROCO_Dacumi

def reetiquetarDataset(dataset, lista_caracteristicas_v1, lista_caracteristicas_v2, df_CROCO_Dacumi, zona, dias_prediccion, dias_caracteristicas):
	columnas = ['Month']

	for i in range(dias_caracteristicas):
		columnas.append('UI_' + str(dias_caracteristicas-1-i))

	for bloques_caracteristicas in lista_caracteristicas_v1:
		for i in range(dias_caracteristicas):
			for car in bloques_caracteristicas:
				columnas.append(car + '_' + str(dias_caracteristicas-1-i))

	for bloques_caracteristicas in lista_caracteristicas_v2:
		for i in range(dias_caracteristicas):
			for car in bloques_caracteristicas:
				columnas.append(car + '_' + str(dias_caracteristicas-1-i))
	dataset.columns = columnas

	dataset = pd.concat([dataset, df_CROCO_Dacumi['Dacuminata'+zona].iloc[dias_prediccion+(dias_caracteristicas-1):].reset_index().rename(columns={'Dacuminata'+zona:'Dacuminata'+zona+'_output'}).iloc[:,1]], axis=1)
	reg_nuls = dataset.isnull().sum()

	for label, content in reg_nuls.items():
		if content > df_CROCO_Dacumi.shape[0] - df_CROCO_Dacumi.shape[0]*0.1:
			dataset.pop(label)
	dataset = dataset.dropna(axis='rows')
	return dataset

def split_datasets(dataset, zona):
	list_temp = []; list_sal = []; list_ui = []; list_dacumi = []; list_u = []; list_v = []; list_w = []; list_bv = []; list_chl =[]
	for elem in dataset:
		if elem[0] == 'T': list_temp.append(elem)
		elif elem[0] == 'S': list_sal.append(elem)
		elif elem[0] == 'U' and elem[1] == 'I': list_ui.append(elem)
		elif elem[0] == 'D' and elem[-1] != 't': list_dacumi.append(elem)
		elif elem[0] == 'U': list_u.append(elem)
		elif elem[0] == 'V': list_v.append(elem)
		elif elem[0] == 'W': list_w.append(elem)
		elif elem[0] == 'C': list_chl.append(elem)
		elif elem[0] == 'B' or elem[0] == 'd': list_bv.append(elem)

	return dataset[list_temp], dataset[list_sal], dataset[list_ui], dataset[list_dacumi], dataset[list_u], dataset[list_v], dataset[list_w], dataset[list_bv], dataset[list_chl], dataset['Month'], dataset['Dacuminata'+zona+'_output']


def regressionModel(x,y, model, args, apply_pca=True):

	if apply_pca:
		pca = PCA(n_components=0.999, svd_solver='full')
		pca.fit(x.iloc[0:365, :])
		x_train = pca.transform(x.iloc[:-365, :])
		y_train = y.iloc[:-365]
		x_test = pca.transform(x.iloc[-365:, :])
		y_test = y.iloc[-365:]
		n_components = pca.n_components_
	else:
		x_train = x.iloc[:-365, :]
		y_train = y.iloc[:-365]
		x_test = x.iloc[-365:, :]
		y_test = y.iloc[-365:]
		n_components = x.shape[1]

	scaler = MinMaxScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	output = []

	if model != 'mlp_bl':
		if model == 'rf_bl':
			regr = RandomForestRegressor(n_estimators=1000, criterion=args[0], max_depth=args[1], random_state=42)

		if model == 'knn_bl':
			regr = KNeighborsRegressor(n_neighbors=args[0])

		if model == 'svr_bl':
			regr = SVR(kernel=args[0], C=args[1], epsilon=args[2], degree=args[3])

		regr.fit(x_train, y_train)
		y_pred = regr.predict(x_test)
		r2 = r2_score(y_test, y_pred)
		mae = mean_absolute_error(y_test, y_pred)
		mse = mean_squared_error(y_test, y_pred)
		rmse = mean_squared_error(y_test, y_pred, squared=False)
		output = y_pred.tolist()
		metric = [r2, mae, mse, rmse]

	if model == 'mlp_bl':

		neuronas_c1 = args[1][0]
		if len(args[1])==2:
			neuronas_c2 = args[1][1]

		inp = x_train.shape[1]

		list_metric_r2 = []
		list_metric_mae = []
		list_metric_mse = []
		list_metric_rmse = []
		for i in range(50):
			modelo = tf.keras.models.Sequential()

			modelo.add(tf.keras.layers.Dense(neuronas_c1, input_dim=inp, activation='linear'))
			if len(args[1]) == 2:
				modelo.add(tf.keras.layers.Dense(neuronas_c2, activation='linear'))
			modelo.add(tf.keras.layers.Dense(1, activation=args[2]))

			optimizer = tf.keras.optimizers.RMSprop(learning_rate=args[0])
			modelo.compile(loss='mse', optimizer=optimizer)

			callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
			modelo.fit(x=x_train,
					   y=y_train,
					   batch_size=128,
					   epochs=1000,
					   validation_split=0.1,
					   validation_freq=1,
					   callbacks=[callback],
					   verbose=0)

			y_pred = modelo.predict(x_test)
			r2 = r2_score(y_test, y_pred)
			mae = mean_absolute_error(y_test, y_pred)
			mse = mean_squared_error(y_test, y_pred)
			rmse = mean_squared_error(y_test, y_pred, squared=False)
			list_metric_r2.append(r2)
			list_metric_mae.append(mae)
			list_metric_mse.append(mse)
			list_metric_rmse.append(rmse)

			if i == 0:
				for idx_testdata in range(len(y_test)):
					output.append(modelo.predict(x_test[idx_testdata, :].reshape(1, -1), verbose=0)[0][0])

			tf.keras.backend.clear_session()
			del modelo

		metric = [np.mean(list_metric_r2), np.mean(list_metric_mae), np.mean(list_metric_mse), np.mean(list_metric_rmse)]

	return metric, output, n_components

def grid_search(dataset, zona, model, i):

	config = {'svr_bl': {'kernel': ['linear', 'rbf'],
					 'c': [0.001, 0.01, 0.05, 0.1, 1, 10],
					 'e': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
			  'mlp_bl': {'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
					'n_neurons': [[8], [10], [2, 2], [4, 2], [10, 10], [10, 2], [16, 8], [32, 16], [32, 8], [64, 32]],
					'activacion_salida':['sigmoid', 'relu']},
			  'rf_bl': {'criterion': ['squared_error', 'friedman_mse', 'poisson'],
					'max_depth': [2, 4, 10, 20, 50, None]}
			  }
	mlp_size = len(config['mlp_bl']['learning_rate']) * len(config['mlp_bl']['n_neurons']) * len(config['mlp_bl']['activacion_salida'])
	rf_size = len(config['rf_bl']['criterion']) * len(config['rf_bl']['max_depth'])
	params = []

	if model == 'svr_bl':
		params = [config['svr_bl']['kernel'][i // (len(config['svr_bl']['c']) * len(config['svr_bl']['e']))],
				  config['svr_bl']['c'][(i // len(config['svr_bl']['e'])) % len(config['svr_bl']['c'])],
				  config['svr_bl']['e'][i % len(config['svr_bl']['e'])],
				  1 , None]
	elif model == 'mlp_bl':
		if i < mlp_size:
			params = [config['mlp_bl']['learning_rate'][i // (len(config['mlp_bl']['n_neurons']) * len(config['mlp_bl']['activacion_salida']))],
					  config['mlp_bl']['n_neurons'][(i // len(config['mlp_bl']['activacion_salida'])) % len(config['mlp_bl']['n_neurons'])],
					  config['mlp_bl']['activacion_salida'][i % (len(config['mlp_bl']['activacion_salida']))],
					  None, None]
	else:
		if i < rf_size:
			params = [config['rf_bl']['criterion'][i // (len(config['rf_bl']['max_depth']))],
					  config['rf_bl']['max_depth'][i % (len(config['rf_bl']['max_depth']))],
					  None, None, None]
	if len(params) == 5:
		dataset_aux = dataset.copy()
		y = dataset_aux.pop('Dacuminata' + zona + '_output')
		X = dataset_aux
		score, output, _ = regressionModel(X, y, model, params, False)

		return score, params, output

	else:
		return None, None, None



