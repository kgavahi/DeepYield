import numpy as np
from pyhdf.SD import SD, SDC
import os
import sys

def MergeTiles(path,DATAFIELD_NAME):
	files = os.listdir(path)
	files = [x for x in files if x.endswith('hdf')]


	path_file = os.path.join(path,files[0])
	hdf = SD(path_file, SDC.READ)
	#for x in hdf.datasets():
	#	print(x)
	_FillValue = hdf.select(DATAFIELD_NAME).getfillvalue()
	data3D = hdf.select(DATAFIELD_NAME)[:]
	cell_size = data3D.shape[0]
	hdf.end()

	print(_FillValue,data3D.dtype)

	Merge = np.zeros([cell_size*3,cell_size*6],dtype=data3D.dtype)
	count = 0
	for i in range(3):
		for j in range(6):
			tile = 'h%02dv%02d'%(j+8,i+4)

			file = [x for x in files if tile in x]
			if file!=[]:
				path_file = os.path.join(path,file[0])
				hdf = SD(path_file, SDC.READ)
				data3D = hdf.select(DATAFIELD_NAME)[:]
				#data3D = np.where(data3D==_FillValue,np.nan,data3D)
				Merge[i*cell_size:(i+1)*cell_size,
				j*cell_size:(j+1)*cell_size] = data3D
				hdf.end()
	
			else:     
				Merge[i*cell_size:(i+1)*cell_size,
				j*cell_size:(j+1)*cell_size] = _FillValue
		
	#np.save('%s/%s'%(path,DATAFIELD_NAME),Merge)		
	return Merge


k = int(sys.argv[1])
Dataset = 'MOD09A1'


dirs = '/mh1/kgavahi/CropYieldProject/MODIS_DATA/%s/'%Dataset
paths = os.listdir(dirs)
paths = [os.path.join(dirs,x) for x in paths]

path = paths[k]



'''  THIS IS FOR MCD12Q1  '''
'''
print(path)
for field in ['LC_Type1','LC_Type2','LC_Type3','LC_Type4']:
		
	Merge_1km = np.ones([3600,7200],dtype='uint8') * 255
		
	DATAFIELD_NAME = field
	print(DATAFIELD_NAME)
	
	Merge = MergeTiles(path,DATAFIELD_NAME) 

	for i in range(3600):
		for j in range(7200):
			window = Merge[i*2:(i+1)*2,j*2:(j+1)*2]

			counts = np.bincount(window.flatten())
			Merge_1km[i,j] = np.argmax(counts)
				
	np.save('%s/%s_1km'%(path,DATAFIELD_NAME),Merge_1km)	
	print(DATAFIELD_NAME,'DONE!')'''
	
	
'''  THIS IS FOR MOD09A1  '''
'''
print(path)
for b in range(1,8):
	Merge_1km = np.ones([3600,7200],dtype='int16')*-28672
	print(Merge_1km.dtype)
	DATAFIELD_NAME = 'sur_refl_b0%d'%b
	print(DATAFIELD_NAME)
	
	Merge = MergeTiles(path,DATAFIELD_NAME) 



	for i in range(3600):
		for j in range(7200):
			window = Merge[i*2:(i+1)*2,j*2:(j+1)*2]
			if (window==-28672).any():
				continue
			Merge_1km[i,j] = np.mean(window)
			print(Merge_1km.dtype)

	np.save('%s/%s_1km'%(path,DATAFIELD_NAME),Merge_1km)
	print(DATAFIELD_NAME,'DONE!')'''
