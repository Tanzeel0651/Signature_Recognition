import os
import numpy as np
import sys
import numpy
from keras.models import model_from_json
from keras.preprocessing import image

def load_predict(i, img_path):
	i = 1
	path = '/home/tanzeel/Documents/Signature_Recognition/model_save/'
	with open(path+'model{}.json'.format(i)) as json_file:
		loaded_model = model_from_json(json_file.read())
	loaded_model.load_weights(path+'model{}.h5'.format(i))

	loaded_model.compile(loss='binary_crossentropy', 
						optimizer='adam', 
						metrics=['accuracy'])

	print('Loaded Model from the disc')

	test_image = image.load_img(img_path, target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	result = loaded_model.predict(test_image)

	if result[0][0] > 0.85:
		#print('Original')
		score = result[0][0]
		#print('Score: ',score)

	else:
		#print('Forged')
		score = 1- result[0][0]
		#print('Score: ',1-score)
	
	return score

if __name__ == '__main__':
	arg = sys.argv[1:]
	score = load_predict(arg[0],arg[1])
	#print(score)

