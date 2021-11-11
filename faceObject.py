class Face:

	# Constructor
	def __init__(self, name, location, encoding):
		self.name = name
		self.location = location
		self.encoding = encoding

	def getName(self):
		return self.name

	def setName(self, name):
		self.name = name

	def getLocation(self):
		return self.location

	def setLocation(self, location):
		self.location = location

	def getEncoding(self):
		return self.encoding

	def setEncoding(self, encoding):
		self.encoding = encoding
