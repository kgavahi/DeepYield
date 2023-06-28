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
def MyResizeFn(x,ToShape):


    #x[:,:,:,:7] = ((x[:,:,:,:7] + 100)/16000)
    #x[:,:,:,7:9] = ((x[:,:,:,7:9] + 12650)/16650)

    x[:,:,:,:7] = ((x[:,:,:,:7] - 100)/(16000-100))
    x[:,:,:,7:9] = ((x[:,:,:,7:9] - 12650)/(16650-12650))
    
    before_1 = np.ceil((ToShape[0] - x.shape[1])/2).astype(int)
    after_1 = ToShape[0] - x.shape[1] - before_1

    before_2 = np.ceil((ToShape[1] - x.shape[2])/2).astype(int)
    after_2 = ToShape[1] - x.shape[2] -  before_2
    
    padded = np.pad(x, ((0,0), (before_1,after_1),(before_2,after_2),(0,0)), 'constant', constant_values=0)

    
    padded = padded[:,:,:,:9]    
    padded = np.where(np.isnan(padded),0,padded)
    padded = np.where(padded<0,0,padded)
    #padded = resize(padded,(64,64,24,9),mode='constant', cval=0)
    return padded
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
			data3D = np.load('../Counties/%s/%s'%(date,county))
			
			mask = np.where(data3D[:,:,9:10]==12,1,0)
			per = np.sum(mask)/(mask.shape[0]*mask.shape[1])

			if per<0.1:
				print('LESS THAN 10%% Cropland')
				break
	
			data3DMasked = data3D * mask
			d3DMaCr = BB(mask[:,:,0], data3DMasked).astype('float32')
			
			if d3DMaCr.shape[0]>image_shape[0] or d3DMaCr.shape[1]>image_shape[1]:
				break
						
			if i==0:
				CONCAT = d3DMaCr.reshape(1, d3DMaCr.shape[0], d3DMaCr.shape[1], d3DMaCr.shape[2])
			else:
				d3DMaCr = d3DMaCr.reshape(1, d3DMaCr.shape[0], d3DMaCr.shape[1], d3DMaCr.shape[2])
				CONCAT = np.append(CONCAT, d3DMaCr, axis=0)
				
		
		try:
			CONCAT = MyResizeFn(CONCAT, image_shape)
			np.save('%s/%d_%s'%(SaveDir, year, county), CONCAT)
		except:
			continue
		
		del CONCAT
	try:
		print(d3DMaCr.shape)
	except:
		pass
		



CountiesUniqueList = np.load('CountiesUniqueList.npy')

image_shape = (70, 160)
#######################################################
SaveDir = '%d_%d_standard'%(image_shape[0], image_shape[1])
os.makedirs(SaveDir, exist_ok=True)
f = pd.read_csv('BuPerAcreAllYears.csv',delimiter=',',low_memory=False)
c=0
#for county in CountiesUniqueList[:1]:
def func(i):
	county = CountiesUniqueList[i]
	s=time.time()
	MaskConcatData(county)	

	print('%.2f'%(time.time()-s),'(s)',c)
	#c+=1



pool = multiprocessing.Pool(processes = 45)
pool.map(func, range(len(CountiesUniqueList)))
pool.close()




