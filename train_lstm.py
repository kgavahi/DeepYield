import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from LoadInputData import LoadData
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.transform import resize
import sys
#########################################################################
f = pd.read_csv('InputData/BuPerAcreAllYears.csv',delimiter=',')
input_folder = 'InputData/70_160_standard'
save_dir = 'training/%s'%sys.argv[1]
os.makedirs(save_dir, exist_ok=True)
image_shape = (70,160)
LoadData(f,input_folder,save_dir)
EPOCHS = 100
batch_size = 32
#########################################################################
def PlotLossAcc(SaveDir):
	history = np.load('%s/history.npy'%SaveDir,allow_pickle=True)
	# summarize history for accuracy
	plt.plot(history.item().get('PCC'))
	plt.plot(history.item().get('val_PCC'))
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	fig = plt.gcf()
	fig.savefig('%s/PCC.png'%SaveDir,dpi=300)
	plt.close()
	# summarize history for loss
	plt.plot(history.item().get('loss'))
	plt.plot(history.item().get('val_loss'))
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.ylim(0,70)
	plt.legend(['train', 'validation'], loc='upper left')
	fig = plt.gcf()
	fig.savefig('%s/loss.png'%SaveDir,dpi=300)
	plt.close()
from keras import backend as K
def PCC(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    #r = K.maximum(K.minimum(r, 1.0), -1.0)
    #return 1 - K.square(r)
    return r
class My_Custom_Generator(tf.keras.utils.Sequence) :
  
	def __init__(self, image_filenames, labels, batch_size) :
		self.image_filenames = image_filenames
		self.labels = labels
		self.batch_size = batch_size
    
    
	def __len__(self) :
		return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
	def __getitem__(self, idx) :
		batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
		batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
		return np.array([
			np.load('%s/'%input_folder + str(file_name))
			for file_name in batch_x]), np.array(batch_y)
def ScatterPlots(i):
	fsize = 14
	Y_pred = model.predict(my_training_batch_generator)
	plt.plot(y_train,Y_pred,'.')
	plt.plot([0,100],[0,100],'-k')
	R = np.corrcoef(y_train[:,0],Y_pred[:,0])[0][1]
	RMSE = np.sqrt(np.mean((y_train[:,0]-Y_pred[:,0])**2))
	plt.xlabel('Observed (million bushels)',fontsize=fsize)
	plt.ylabel('Predicted (million bushels)',fontsize=fsize)
	plt.title('R = %.2f, RMSE = %.3f'%(R,RMSE),fontsize=fsize)
	fig = plt.gcf()
	fig.set_size_inches(4, 4) #width, Height
	fig.savefig('%s/train_scatter_%d.png'%(save_dir,i),dpi=300)
	plt.close()'''

	'''Y_pred = model.predict(my_validation_batch_generator)
	plt.plot(y_val,Y_pred,'.')
	plt.plot([0,100],[0,100],'-k')
	R = np.corrcoef(y_val[:,0],Y_pred[:,0])[0][1]
	RMSE = np.sqrt(np.mean((y_val[:,0]-Y_pred[:,0])**2))
	plt.xlabel('Observed (million bushels)',fontsize=fsize)
	plt.ylabel('Predicted (million bushels)',fontsize=fsize)
	plt.title('R = %.2f, RMSE = %.3f'%(R,RMSE),fontsize=fsize)
	fig = plt.gcf()
	fig.set_size_inches(4, 4) #width, Height
	fig.savefig('%s/val_scatter_%d.png'%(save_dir,i),dpi=300)
	plt.close()
	
	Y_pred = model.predict(my_test_batch_generator)
	plt.plot(y_test,Y_pred,'.')
	plt.plot([0,100],[0,100],'-k')
	R = np.corrcoef(y_test[:,0],Y_pred[:,0])[0][1]
	RMSE = np.sqrt(np.mean((y_test[:,0]-Y_pred[:,0])**2))
	print(i, R, RMSE)
	plt.xlabel('Observed (million bushels)',fontsize=fsize)
	plt.ylabel('Predicted (million bushels)',fontsize=fsize)
	plt.title('R = %.2f, RMSE = %.3f'%(R,RMSE),fontsize=fsize)
	fig = plt.gcf()
	fig.set_size_inches(4, 4) #width, Height
	fig.savefig('%s/test_scatter_%d.png'%(save_dir,i),dpi=300)
	plt.close()
	return R, RMSE

filenames = np.load('%s/filenames.npy'%save_dir)
labels = np.load('%s/y_labels.npy'%save_dir)

filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels, random_state=1)

X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
    filenames_shuffled, y_labels_shuffled, test_size=0.2, random_state=1)

#X_test_filenames, X_val_filenames, y_test, y_val = train_test_split(
#    X_val_filenames, y_val, test_size=0.5, random_state=1)
X_test_filenames = np.load('%s/filenames_test.npy'%save_dir)
y_test = np.load('%s/y_labels_test.npy'%save_dir)

my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size)		
my_test_batch_generator = My_Custom_Generator(X_test_filenames, y_test, batch_size)	

def create_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.ConvLSTM2D(filters=10, kernel_size=(3, 3),
			input_shape=(24, 70, 160, 9),
			padding='same', return_sequences=True),
		tf.keras.layers.ConvLSTM2D(filters=8, kernel_size=(3, 3),
			padding='same', return_sequences=True),
		tf.keras.layers.ConvLSTM2D(filters=6, kernel_size=(3, 3),
			padding='same', return_sequences=True),
		tf.keras.layers.ConvLSTM2D(filters=5, kernel_size=(3, 3),
			padding='same', return_sequences=True),
		tf.keras.layers.ConvLSTM2D(filters=4, kernel_size=(3, 3),
			padding='same', return_sequences=True),
		tf.keras.layers.ConvLSTM2D(filters=3, kernel_size=(3, 3),
			padding='same', return_sequences=True),
		tf.keras.layers.ConvLSTM2D(filters=2, kernel_size=(3, 3),
			padding='same', return_sequences=True),
		tf.keras.layers.ConvLSTM2D(filters=1, kernel_size=(3, 3),
			padding='same', return_sequences=False),
		tf.keras.layers.BatchNormalization(),
		#tf.keras.layers.Conv3D(1, (3,3,3),padding='same',data_format='channels_last', activation='relu'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512, activation='linear'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(1, activation='linear')
	])


	model.compile(optimizer="adam", loss='mean_squared_error', metrics=[PCC])
	
	return model

checkpoint_path = '%s/cp-{epoch:04d}.ckpt'%save_dir
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
												save_weights_only=True,
												verbose=1)
model = create_model()
print(model.summary())


history = model.fit_generator(generator=my_training_batch_generator,
                   steps_per_epoch = int(len(X_train_filenames) // batch_size),
                   epochs = EPOCHS,
                   verbose = 1,
                   validation_data = my_validation_batch_generator,
                   validation_steps = int(len(X_val_filenames) // batch_size),
				   callbacks=[cp_callback])
np.save('%s/history'%save_dir,history.history)

R = []
RMSE = []
for i in range(1,EPOCHS+1):
	model.load_weights('%s/cp-%04d.ckpt'%(save_dir,i))

	pcc, rmse = ScatterPlots(i)
	R.append(pcc)
	RMSE.append(rmse)
print('mean R:', np.mean(R), 'mean RMSE:', np.mean(RMSE), 'min RMSE:', np.min(RMSE))
np.save('%s/R_test'%save_dir,R)
np.save('%s/RMSE_test'%save_dir,RMSE)
PlotLossAcc(save_dir)

history = np.load('%s/history.npy'%save_dir,allow_pickle=True)
x = history.item().get('val_loss')
print('epoch:', np.argmin(x)+1, 'RMSE:', np.min(x))




