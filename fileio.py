import pickle
import os
class FileIO:
	def save_obj(self, obj, name):
		filename = os.path.join('C:/workspace/', name + '.pkl')
		with open(filename, 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def load_obj(self, name):
		filename = os.path.join('C:/workspace/', name + '.pkl')
		with open(filename, 'rb') as f:
			return pickle.load(f)

	def check_obj(self, name):
		filename = os.path.join('C:/workspace/', name + '.pkl')
		return os.path.exists(filename)