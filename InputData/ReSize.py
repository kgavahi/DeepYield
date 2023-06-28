import numpy as np
from skimage.transform import resize
import os
import multiprocessing
def MyResizeFn(file_name):

    x = np.load('%s/'%input_folder + str(file_name))
    x[:,:,:,:7] = ((x[:,:,:,:7] + 100)/16000)
    x[:,:,:,7:9] = ((x[:,:,:,7:9] + 12650)/16650)
    
    before_1 = np.ceil((ToShape[0] - x.shape[0])/2).astype(int)
    after_1 = ToShape[0] - x.shape[0] - before_1

    before_2 = np.ceil((ToShape[1] - x.shape[1])/2).astype(int)
    after_2 = ToShape[1] - x.shape[1] -  before_2
    
    padded = np.pad(x, ((before_1,after_1),(before_2,after_2),(0,0),(0,0)), 'constant', constant_values=0)
    #padded = resize(x,(64,64,24,10),mode='constant', cval=0)
    
    padded = padded[:,:,:,:9]    
    padded = np.where(np.isnan(padded),0,padded)
    padded = resize(padded,(64,64,24,9),mode='constant', cval=0)
    np.save('70_160_resize/'+file_name,padded)
    #return padded
input_folder = '70_160'
ToShape = (70,160)	
files = os.listdir('70_160')


pool = multiprocessing.Pool(processes = 45)
pool.map(MyResizeFn, files)
pool.close()
