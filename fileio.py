import pickle
import os
class FileIO:
	def save_obj(self, obj, name):
		with open(name + '.pkl', 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def load_obj(self, name):
		with open(name + '.pkl', 'rb') as f:
			return pickle.load(f)

	def check_obj(self, name):
		return os.path.exists(name+'.pkl')