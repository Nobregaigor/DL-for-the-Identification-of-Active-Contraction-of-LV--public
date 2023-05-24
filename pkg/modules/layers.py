import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense

class ResnetBlock(tf.keras.Model):
  def __init__(self, units, dropout_rate=0.001, **kwargs):
    super(ResnetBlock, self).__init__(**kwargs)

    self.units=units
    self.dropout_rate=dropout_rate
    self.dense1 = Dense(units)
    self.dense2 = Dense(units)
    self.dense3 = Dense(units)

    self.dropout = Dropout(dropout_rate)
  
  @tf.function()
  def call(self, input_tensor, training=False):
    x = self.dense1(input_tensor, training=training)
    x = self.dropout(x)
    x = self.dense2(x, training=training)
    x = self.dropout(x)
    x = self.dense3(x, training=training)
    x = self.dropout(x)

    x += input_tensor
    return tf.nn.relu(x)
  
  def get_config(self):
      config = super(ResnetBlock, self).get_config()
      config.update({"units": self.units, "dropout_rate": self.dropout_rate})
      return config