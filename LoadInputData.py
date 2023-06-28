import pandas as pd
import numpy as np
import os

def LoadData(f,input_folder,save_dir):
	files = os.listdir(input_folder)
	m = len(files)
	
	filenames = []
	labels = np.empty((0, 1),dtype='float64')

	filenames_test = []
	labels_test = np.empty((0, 1),dtype='float64')

	count=0
	for name in files:
		print(count)

		year = name.split('_')[0]
		statefp = name.split('_')[2]
		countyfp = name.split('_')[3]
		if year!='2019' and year!='2018':
			filenames.append(name)
		else:
			filenames_test.append(name)
		
		

		
		f_year = f.loc[f.Year==int(year)]
		f_year = f_year.loc[f_year.Program=='SURVEY']
		f_county = f_year.loc[f_year['State ANSI']==int(statefp)]
		f_county = f_county.loc[f_county['County ANSI']==int(countyfp)]
		
		#Yield = int(f_county.Value.values[0].replace(',',''))
		Yield = f_county.Value.values[0]
		
		#labels[count,0] = Yield/1000000
		if year!='2019' and year!='2018':
			labels = np.append(labels,[[Yield]],axis=0)
		else:
			labels_test = np.append(labels_test,[[Yield]],axis=0)
		count+=1
		

	np.save('%s/filenames.npy'%save_dir, filenames)
	np.save('%s/y_labels.npy'%save_dir, labels)
	np.save('%s/filenames_test.npy'%save_dir, filenames_test)
	np.save('%s/y_labels_test.npy'%save_dir, labels_test)