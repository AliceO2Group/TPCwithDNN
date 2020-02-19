import keras.backend as K
from keras.layers import Layer

class SymmetricPadding3D(Layer):
	def __init__(self,   padding=[[0,0],[1,1],[1,1],[1,1],[0,0]],mode = 'SYMMETRIC', data_format="channels_last", **kwargs):
		self.data_format = data_format
		self.padding = padding
		self.mode = mode
		super(SymmetricPadding3D, self).__init__(**kwargs)
	
	def build(self, input_shape):
		super(SymmetricPadding3D, self).build(input_shape)

	def call(self, inputs):
		#pad = self.padding
		#if self.data_format is "channels_last":
		#(batch, depth, rows, cols, channels
		pad = [[0,0]] + [i for i in self.padding] + [[0,0]]
		#elif self.data_format is "channels_first":
            #(batch, channels, depth, rows, cols)
		#	pad = [[0, 0], [0, 0]] + [i for i in self.padding]

	
		if K.backend() == "tensorflow":
			import tensorflow as tf
			paddings = tf.constant(pad)
			print(inputs.shape)
			print(paddings)
			out = tf.pad(inputs, paddings, self.mode)
		else:
			raise Exception("Backend " + K.backend() + "not implemented")
		#self.output_dim  = [ (out.shape[0].value , out.shape[1].value, out.shape[2].value, out.shape[3].value, out.shape[4].value)]
		self.output_dim  = [ (out.shape[0] , out.shape[1], out.shape[2], out.shape[3], out.shape[4])]
		return out 
	
	def compute_output_shape(self,input_shape):
		return  self.output_dim

	def  get_config(self):
		config = {'padding': self.padding,
                  	  'data_format': self.data_format,
			  'mode': self.mode}
		base_config = super(SymmetricPadding3D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
    
