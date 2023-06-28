import numpy as np
import os
import sys
import copy
import datetime
import pandas as pd
import time
import multiprocessing
def timeDeltaWalk(date, delta):
	date = datetime.date(
	int(str(date)[ :4]), 
	int(str(date)[4:6]), 
	int(str(date)[6:8]))
	date = str(date + datetime.timedelta(days = delta))
	return date	
def BB(mask,data):
	data_new = copy.deepcopy(data)
	N = mask.shape[0]
	M = mask.shape[1]
	for i in range(N):
		if np.sum(mask[i,:])==0:
			I1 = i
		else:
			I1 = i
			break
	for i in range(N):
		if np.sum(mask[N-i-1,:])==0:
			I2 = N-i-1
		else:
			if i==0:
				I2 = N-i
			break


	for j in range(M):
		if np.sum(mask[:,j])==0:
			J1 = j
		else:
			J1 = j
			break
	for j in range(M):
		if np.sum(mask[:,M-j-1])==0:
			J2 = M-j-1
		else:
			if j==0:
				J2 = M-j
			break
	data_new = data[I1:I2,J1:J2,:]
	return data_new	
def MaskConcatData(county):
	for year in range(2003,2020):
		f_year = f.loc[f.Year==year]
		f_year = f_year.loc[f_year.Program=='SURVEY']
		CountySplit = county.split('_')
		f_county = f_year.loc[f_year['State ANSI']==int(CountySplit[1])]
		f_county = f_county.loc[f_county['County ANSI']==int(CountySplit[2])]
		try:
			#Yield = int(f_county.Value.values[0].replace(',',''))
			Yield = f_county.Value.values[0]
		except:
			continue
		#print(year)
		for i in range(24):
			date = timeDeltaWalk('%d0218'%year, 8*(i))
			data3D = np.load('Counties/%s/%s'%(date,county))
			
			mask = np.where(data3D[:,:,9:10]==12,1,0)
			per = np.sum(mask)/(mask.shape[0]*mask.shape[1])

			if per<0.1:
				print('LESS THAN 10%% Cropland')
				break
	
			data3DMasked = data3D * mask
			d3DMaCr = BB(mask[:,:,0],data3DMasked).astype('float32')
			
			if d3DMaCr.shape[0]>image_shape[0] or d3DMaCr.shape[1]>image_shape[1]:
				break
						
			if i==0:
				CONCAT = d3DMaCr.reshape(d3DMaCr.shape[0],d3DMaCr.shape[1],1,d3DMaCr.shape[2])
			else:
				d3DMaCr = d3DMaCr.reshape(d3DMaCr.shape[0],d3DMaCr.shape[1],1,d3DMaCr.shape[2])
				CONCAT = np.append(CONCAT,d3DMaCr,axis=2)
				
		
		try:
			np.save('InputData/%d_%d/%d_%s'%(image_shape[0],image_shape[1],year,county),CONCAT)
		except:
			continue
		
		del CONCAT
	try:
		print(d3DMaCr.shape)
	except:
		pass
		

		#CHECK = np.nansum(np.load('test/%d_%s'%(year,county))-np.load('InputData/%d_%d/%d_%s'%(image_shape[0],image_shape[1],year,county)))
		#print(CHECK)
		#if CHECK!=0:
		#	print('WOWOWOWOWOWOWOWOWOWOW')

#a = int(sys.argv[1])
CountiesUniqueList = np.load('CountiesUniqueList.npy')
#CountiesUniqueList_2 = CountiesUniqueList[2*a:2*a+2]
#print(CountiesUniqueList_2)
image_shape = (1000,1000)
#######################################################

os.makedirs('InputData/%d_%d'%(image_shape[0],image_shape[1]) ,exist_ok=True)
f = pd.read_csv('BuPerAcreAllYears.csv',delimiter=',',low_memory=False)
c=0
#for county in CountiesUniqueList_2:
def func(i):
	county = CountiesUniqueList[i]
	s=time.time()
	MaskConcatData(county)	
	#c+=1
	print('%.2f'%(time.time()-s),'(s)',c)



pool = multiprocessing.Pool(processes = 45)
pool.map(func, range(len(CountiesUniqueList)))
pool.close()




