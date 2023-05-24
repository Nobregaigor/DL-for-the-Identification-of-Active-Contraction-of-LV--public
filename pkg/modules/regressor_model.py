import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from .layers import ResnetBlock

# -------------------------------------------------------------------
# BIN Layer Model

# -------------------------------------------------------------------
# BIN Layer Model

class BMProcessLayer(tf.keras.layers.Layer):
  def __init__(self, n_classes, n_units_connection, **kwargs):
    super(BMProcessLayer, self).__init__(**kwargs)
    
    self.n_classes = n_classes
    self.n_units_connection = n_units_connection

    self.l1 = ResnetBlock(n_units_connection, dropout_rate=0.0001)
    self.l2 = Dropout(0.01)
    self.l3 = Dense(n_classes+1, activation="softmax")

  def call(self, input_tensor, training=False):
    x = self.l1(input_tensor, training=training)
    x = self.l2(x, training=training)
    x = self.l3(x, training=training)
    return x
  
  def get_config(self):
      config = super(BMProcessLayer, self).get_config()
      config.update({"n_classes": self.n_classes, 
                     "n_units_connection": self.n_units_connection})
      return config

class BM_Model(tf.keras.Model):
  def __init__(self, bm_norm, bm_main, n_classes, **kwargs):
    super(BM_Model, self).__init__(**kwargs)
    
    self.n_classes = n_classes
    self.main = bm_main
    self.norm = bm_norm

  def call(self, input_tensor, training=False):
    x = self.norm(input_tensor)
    x = self.main(x)
    x = tf.math.argmax(x, axis=-1)
    return x
  
  def get_config(self):
      config = super(BM_Model, self).get_config()
      config.update({"n_classes": self.n_classes})
      return config 

class BM_Model_Normalized(tf.keras.layers.Layer):
  def __init__(self, bm_norm, bm_main, n_classes, **kwargs):
    super(BM_Model_Normalized, self).__init__(**kwargs)
    
    self.n_classes = n_classes
    self.main = bm_main
    self.norm = bm_norm

  def call(self, input_tensor, training=False):
    x = self.norm(input_tensor)
    x = self.main(x)
    x = tf.math.argmax(x, axis=-1)
    x = tf.cast(x, tf.float32)
    x = tf.math.divide_no_nan(x, [20.0, 20.0, 50.0, 50.0])
    return x
  
  def get_config(self):
      config = super(BM_Model_Normalized, self).get_config()
      config.update({"n_classes": self.n_classes})
      return config

def build_bm_norm_layer(batch_size, header, labels, feats, name="bm_norm"):
  from tensorflow.keras.layers.experimental import preprocessing
  ds_norm  = csv_ds_reader(train_dir, header, labels, feats, int_labels=True, batch_size=batch_size)
  norm_layer = preprocessing.Normalization(name=name)
  for (xs, _) in ds_norm.take(1):
    norm_layer.adapt(xs)
  return norm_layer

def build_bm_main(n_classes, n_units, n_units_connection, n_input_feats,prefix,
    ):

  bm_in = layers.Input([n_input_feats], name=prefix+"_bin_main")
  d1 = Dense(n_units, activation="swish", name="bm_common")(bm_in)
  d2 = Dense(n_units_connection, activation="sigmoid", name="bm_connector")(d1)
    
  bm_blocks = []
  for name in BIN_LABELS:
    bm_blocks.append(
        BMProcessLayer(
            n_classes, 
            n_units_connection, 
            name="{}_in".format(name)
            )
        )
  bm_conn = []
  for block in bm_blocks:
    bm_conn.append(block(d2))

  bm_out = layers.Lambda(lambda x: tf.stack(x, axis=1), name=prefix+"bin_out")(bm_conn)
  bm_main = tf.keras.Model(inputs=[bm_in], outputs=[bm_out], name=prefix+"bin")
  bm_main.summary()
  return bm_main

# -------------------------------------------------------------------
# GM4 Regressor Model

def clone_norm_layer(batch_size, name, NormLayer):
  from tensorflow.keras.layers.experimental import preprocessing

  ds_norm_batch  = csv_ds_reader(train_dir, BIN_LABELS, FEATS, int_labels=True, batch_size=batch_size)
  normlayer = preprocessing.Normalization(name=name)

  for (xs, _) in ds_norm_batch.take(1):
    normlayer.adapt(xs)
  
  normlayer.set_weights(NormLayer.get_weights())
  return normlayer

class GM4_Regressor_Model(tf.keras.Model):
  def __init__(self, n_units, n_blocks, bm_model_normalized, n_output_labels, **kwargs):
    super(GM4_Regressor_Model, self).__init__(**kwargs)

    self.n_units = n_units
    self.n_blocks = n_blocks
    self.norm = clone_norm_layer(bm_model_normalized.norm)
    self.bins = bm_model_normalized

    self.d1 = Dense(self.n_units, activation="relu")
    self.l1 = [ResnetBlock(self.n_units, dropout_rate=0.001) for _ in range(n_blocks)]
    self.l2 = [ResnetBlock(self.n_units*2, dropout_rate=0.001) for _ in range(n_blocks)]
    self.l3 = [ResnetBlock(self.n_units*4, dropout_rate=0.001) for _ in range(n_blocks)]

    self.d2 = Dense(self.n_units*2, activation="relu")
    self.d3 = Dense(self.n_units*4, activation="relu")

    self.tail = [
      layers.ActivityRegularization(0.0001, 0.0001),
      Dense(n_output_labels, name="gr_body_out"),
    ]

  def call(self, input_tensor, training=False):
    x_norm = self.norm(input_tensor)
    x_bins = self.bins(input_tensor)
    x = tf.concat([x_norm, x_bins], axis=1)
    x = self.d1(x)
    for layer in self.l1:
      x = layer(x)
    x = self.d2(x)
    x = self.d3(x)
    for layer in self.tail:
      x = layer(x)
    return x

  def get_config(self):
      # config = super(CM4_Model, self).get_config()
      return {"n_units": self.n_units, "n_blocks": self.n_blocks}
