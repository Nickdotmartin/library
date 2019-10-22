import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers import GaussianNoise

from keras import backend as K

"""
/home/nm13850/Documents/PhD/Python/learning_new_functions/CNN_sim_script/conv_march_2019/conv_tutorial3/train_vgg.py
To use the model call it somethig like this...
# initialize our VGG-like Convolutional Neural Network
model = con6_pool3_fc1.build(width=64, height=64, depth=3, classes=len(lb.classes_))

# initialize our initial learning rate, # of epochs to train for, and batch size
INIT_LR = 0.01
EPOCHS = 75
BS = 32

# initialize the model and optimizer (you'll want to use binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
It may be appropriate before the activation function for activations that may result in non-Gaussian distributions 
like the rectified linear activation function, the modern default for most network types.
"The goal of Batch Normalization is to achieve a stable distribution of activation values throughout training, 
and in our experiments we apply it before the nonlinearity since that is where matching the first and second moments 
is more likely to result in a stable distribution" 
	— Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 2015.
"""


# todo: replace fc layers with global average pooling layers.
#  Possibly this will require a global pooling layers with n_pooling kernels =  n_cov_filters from previous layers.
#  Still have a dense layer (with n_units = n_cats) but now the number of parameters is less than if I used average pooling?

# todo: 1x1 conv filters?

# todo: move bn and dropout at end to BEFORE fc, not after!

class con6_pool3_fc1:
	@staticmethod
	def build(width, height, depth, classes, batch_norm=True, dropout=True):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential(layers=6, name="con6_pool3_fc1")
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu', name='conv_1'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_1'))
		# model.add(Activation("relu", name='activation_1'))
		model.add(MaxPooling2D(pool_size=(2, 2), name='mpool_1'))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_1'))

		# CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same", name='conv_2', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_2'))
		# model.add(Activation("relu", name='activation_2'))
		model.add(Conv2D(64, (3, 3), padding="same", name='conv_3', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_3'))
		# model.add(Activation("relu", name='activation_3'))
		model.add(MaxPooling2D(pool_size=(2, 2), name='mpool_2'))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_2'))

		# CONV => RELU => CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(128, (3, 3), padding="same", name='conv_4', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_4'))
		# model.add(Activation("relu", name='activation_4'))
		model.add(Conv2D(128, (3, 3), padding="same", name='conv_5', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_5'))
		# model.add(Activation("relu", name='activation_5'))
		model.add(Conv2D(128, (3, 3), padding="same", name='conv_6', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_6'))
		# model.add(Activation("relu", name='activation_6'))
		model.add(MaxPooling2D(pool_size=(2, 2), name='mpool_3'))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_3'))

		# first (and only) set of FC => RELU layers
		model.add(Flatten(name='flatten_1'))
		model.add(Dense(512, name='fc_1', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_7'))
		# model.add(Activation("relu", name='activation_7'))
		if dropout is True:
			model.add(Dropout(0.5, name='dropout_4'))

		# softmax classifier
		model.add(Dense(classes, name='Output_fc', activation='softmax'))
		# model.add(Activation("softmax", name='softmax'))

		# return the constructed network architecture
		return model


class con4_pool2_fc1:
	"""
	based on https://keras.io/examples/cifar10_cnn/
	Train a simple deep CNN on the CIFAR10 small images dataset.
	It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
	(it's still underfitting at that point, though).
	"""
	@staticmethod
	def build(width, height, depth, classes, batch_norm=True, dropout=True):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential(layers=4, name="con4_pool2_fc1")
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# first set of CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, name='conv_1', activation='relu'))
		# model.add(Activation("relu", name='activation_1'))
		model.add(Conv2D(32, (3, 3), padding="same", name='conv_2', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_1'))
		# model.add(Activation("relu", name='activation_2'))
		model.add(MaxPooling2D(pool_size=(2, 2), name='mpool_1'))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_1'))

		# second set of CONV => RELU => CONV => RELU => POOL layers
		model.add(Conv2D(64, (3, 3), padding="same", name='conv_3', activation='relu'))
		# model.add(Activation("relu", name='activation_3'))
		model.add(Conv2D(64, (3, 3), padding="same", name='conv_4', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_2'))
		# model.add(Activation("relu", name='activation_4'))
		model.add(MaxPooling2D(pool_size=(2, 2), name='mpool_2'))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_2'))

		# first (and only) set of FC => RELU layers
		model.add(Flatten(name='flatten_1'))
		model.add(Dense(512, name='fc_1', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_3'))
		# model.add(Activation("relu", name='activation_5'))
		if dropout is True:
			model.add(Dropout(0.5, name='dropout_3'))

		# softmax classifier
		model.add(Dense(classes, name='Output_fc', activation='softmax'))
		# model.add(Activation("softmax", name='softmax'))

		# return the constructed network architecture
		return model



class con2_pool2_fc1:
	@staticmethod
	def build(width, height, depth, classes, batch_norm=True, dropout=True):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential(layers=2, name="con2_pool2_fc1")
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, name='conv_1', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_1'))
		# model.add(Activation("relu", name='activation_1'))
		model.add(MaxPooling2D(pool_size=(2, 2), name='mpool_1'))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_1'))

		# CONV => RELU => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same", name='conv_2', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_2'))
		# model.add(Activation("relu", name='activation_2'))
		model.add(MaxPooling2D(pool_size=(2, 2), name='mpool_2'))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_2'))

		# first (and only) set of FC => RELU layers
		model.add(Flatten(name='flatten_1'))
		model.add(Dense(512, name='fc_1', activation='relu'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_3'))
		# model.add(Activation("relu", name='activation_3'))
		if dropout is True:
			model.add(Dropout(0.5, name='dropout_3'))

		# softmax classifier
		model.add(Dense(classes, name='Output_fc', activation='softmax'))
		# model.add(Activation("softmax", name='softmax'))

		# return the constructed network architecture
		return model


class con4_pool2_fc1_reluconv:
	"""
	based on https://keras.io/examples/cifar10_cnn/
	Train a simple deep CNN on the CIFAR10 small images dataset.
	It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
	(it's still underfitting at that point, though).
	just truing different format - adding relu to conv ;layers rather than after

	it makes no difference
	"""
	@staticmethod
	def build(width, height, depth, classes, batch_norm=True, dropout=True):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential(layers=2, name="con4_pool2_fc1_reluconv")
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# first set of CONV => CONV => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu', name='conv_1'))
		model.add(Conv2D(32, (3, 3), activation='relu', name='conv_2'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_1'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="mpool_1"))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_1'))

		# second set of CONV => CONV => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same",  activation='relu', name='conv_3'))
		model.add(Conv2D(64, (3, 3), activation='relu', name='conv_4'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_2'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='mpool_2'))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_2'))

		# first (and only) FC layer
		model.add(Flatten())
		model.add(Dense(512, activation='relu', name='fc_1'))
		if batch_norm is True:
			model.add(BatchNormalization(name='bn_3'))
		if dropout is True:
			model.add(Dropout(0.5, name='dropout_3'))

		# softmax classifier
		model.add(Dense(classes, name='Output_fc', activation='softmax'))
		# model.add(Activation("softmax", name='softmax'))

		# return the constructed network architecture
		return model

class con4_pool2_fc1_noise_layer:
	"""
	based on https://keras.io/examples/cifar10_cnn/
	Train a simple deep CNN on the CIFAR10 small images dataset.
	It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
	(it's still underfitting at that point, though).
	"""
	@staticmethod
	def build(width, height, depth, classes, batch_norm=True, dropout=True):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential(layers=4, name="con4_pool2_fc1_noise_layer")
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# first set of NOISE => CONV (RELU) => NOISE => CONV (RELU) => POOL layer set
		model.add(GaussianNoise(stddev=0.1, input_shape=inputShape, name='noise_1'))
		model.add(Conv2D(32, (3, 3), padding="same", activation='relu', name='conv_1'))
		model.add(GaussianNoise(stddev=0.1, name='noise_2'))
		model.add(Conv2D(32, (3, 3), activation='relu', name='conv_2'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_1'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="mpool_1"))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_1'))

		# second set of NOISE => CONV (RELU) => NOISE => CONV (RELU) => POOL
		model.add(GaussianNoise(stddev=0.1, name='noise_3'))
		model.add(Conv2D(64, (3, 3), padding="same",  activation='relu', name='conv_3'))
		model.add(GaussianNoise(stddev=0.1, name='noise_4'))
		model.add(Conv2D(64, (3, 3), activation='relu', name='conv_4'))
		if batch_norm is True:
			model.add(BatchNormalization(axis=chanDim, name='bn_2'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='mpool_2'))
		if dropout is True:
			model.add(Dropout(0.25, name='dropout_2'))

		# first (and only) set of NOISE => FC (RELU) layers
		model.add(Flatten())
		model.add(GaussianNoise(stddev=0.1, name='noise_5'))
		model.add(Dense(512, activation='relu', name='fc_1'))
		if batch_norm is True:
			model.add(BatchNormalization(name='bn_3'))
		if dropout is True:
			model.add(Dropout(0.5, name='dropout_3'))

		# softmax classifier
		model.add(GaussianNoise(stddev=0.1, name='noise_6'))
		model.add(Dense(classes, name='Output_fc', activation='softmax'))
		# model.add(Activation("softmax", name='softmax'))

		# return the constructed network architecture
		return model