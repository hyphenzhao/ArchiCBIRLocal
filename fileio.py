import pickle
import os
import sys
class FileIO:
	def __init__(self):
		if getattr(sys, 'frozen', False):
			application_path = os.path.dirname(sys.executable)
		elif __file__:
			application_path = os.path.dirname(__file__)
		self.application_path = application_path
	def save_obj(self, obj, name):
		filename = os.path.join(self.application_path, name + '.pkl')
		with open(filename, 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def load_obj(self, name):
		filename = os.path.join(self.application_path, name + '.pkl')
		with open(filename, 'rb') as f:
			return pickle.load(f)

	def check_obj(self, name):
		filename = os.path.join(self.application_path, name + '.pkl')
		return os.path.exists(filename)