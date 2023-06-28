import numpy as np
import os
import time
import shutil
import sys




dirs = '/mh1/kgavahi/CropYieldProject/MODIS_DATA/MOD09A1'
dates = os.listdir(dirs)

dirs = './index_county'
paths = os.listdir(dirs)
paths_ind = [os.path.join(dirs,x) for x in paths]



Datasets = ['MOD09A1','MYD11A2','MCD12Q1']
SDS_NAMES = [['sur_refl_b01_1km','sur_refl_b02_1km',
'sur_refl_b03_1km','sur_refl_b04_1km',
'sur_refl_b05_1km','sur_refl_b06_1km',
'sur_refl_b07_1km'],['LST_Day_1km','LST_Night_1km'],['LC_Type2_1km']]

SDS_Fill = [-28672,0,255]

date = dates[int(sys.argv[1])]
print(date)
os.mkdir('./Counties/%s'%date)
s = time.time()
for county in paths_ind:
	print(county)
	ind = np.genfromtxt(county,delimiter=',').astype(int)
	ind = np.where(ind<0,0,ind)
		
	data_county = np.zeros([ind.shape[0],ind.shape[1],10])
	b=0
	for D in range(3):
		dir = '/mh1/kgavahi/CropYieldProject/MODIS_DATA/%s/%s'%(Datasets[D],date)
		if D==2:
			dir = '/mh1/kgavahi/CropYieldProject/MODIS_DATA/%s/%s-01-01'%(Datasets[D],date[:4])
			if not os.path.exists(dir):
				dir = '/mh1/kgavahi/CropYieldProject/MODIS_DATA/%s/%s-01-01'%(Datasets[D],'2018')
		for S in SDS_NAMES[D]:
			#print(Datasets[D],S)
			data3D = np.load('%s/%s.npy'%(dir,S))
			data3D = np.where(data3D==SDS_Fill[D],np.nan,data3D)
			data3D = data3D.flatten()
			data3D[0] = np.nan
				

			data_county[:,:,b] = data3D[ind]
			b+=1
					
	np.save('./Counties/%s/%s'%(date,county[14:-4]),data_county)

print((time.time()-s),'(s)')
		


