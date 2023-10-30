import os
import pickle

class GetResources:

	@staticmethod
	def getModel():
		model_file = 'model/house.pickle'
		file = open(model_file, 'rb')
		model = pickle.load(file)
		file.close()
		return model

	@staticmethod
	def getLookupDict():
		dict_file =  'dict/labelDict.pickle'
		file = open(dict_file,'rb')
		lookupDict = pickle.load(file)
		file.close()
		return lookupDict
