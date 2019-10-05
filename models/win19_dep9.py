from keras.layers import Conv2D, BatchNormalization
from keras.models import Sequential

def CreateNetwork(input, inputShape, scope='win19_dep9'):
	numMaps = 64 # Tamanho do imagem de saida
	kw = 3 # Numero de colunas do kernel
	kh = 3 # Numero de linahs do kernel

	model = Conv2D(numMaps, (kw, kh), input_shape=inputShape, padding='valid', activation='relu')(input)
	model = BatchNormalization()(model)

	model = Conv2D(numMaps, (kw, kh), padding='valid', activation='relu')(model)
	model = BatchNormalization()(model)

	model = Conv2D(numMaps, (kw, kh), padding='valid', activation='relu')(model)
	model = BatchNormalization()(model)

	model = Conv2D(numMaps, (kw, kh), padding='valid', activation='relu')(model)
	model = BatchNormalization()(model)

	model = Conv2D(numMaps, (kw, kh), padding='valid', activation='relu')(model)
	model = BatchNormalization()(model)

	model = Conv2D(numMaps, (kw, kh), padding='valid', activation='relu')(model)
	model = BatchNormalization()(model)

	model = Conv2D(numMaps, (kw, kh), padding='valid', activation='relu')(model)
	model = BatchNormalization()(model)

	model = Conv2D(numMaps, (kw, kh), padding='valid', activation='relu')(model)
	model = BatchNormalization()(model)

	model = Conv2D(numMaps, (kw, kh), padding='valid')(model)
	model = BatchNormalization()(model)

	return model
