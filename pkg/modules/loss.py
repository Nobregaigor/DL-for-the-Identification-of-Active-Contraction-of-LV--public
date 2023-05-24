import tensorflow as tf

class NormalizedBaseClass(tf.keras.losses.Loss):
  def __init__(self, lb=[], up=[], indices=[], epsilon=1e-12, dtype=tf.float32, **kwargs):
    super().__init__(**kwargs)

    self.indices = indices
    self.use_indices = True if len(indices) > 0 else False

    self.epsilon = epsilon
    self.dtype = dtype

    self.lowBound = self.setBounds(lb)
    self.uppBound = self.setBounds(up)
    self.epsilon_tensor = self.setEpsilonTensor(epsilon)
    
    self._dividion_denomiator = tf.maximum(tf.subtract(self.uppBound, self.lowBound, name="denom_sub"), self.epsilon_tensor)    
  
  def setBounds(self, bd):
    return tf.constant(self.gather(bd), dtype=self.dtype)
  
  def setEpsilonTensor(self,epsilon):
    return tf.constant(self.epsilon, dtype=self.dtype, shape=self.lowBound.shape)

  def gather(self, tensor, axis=-1):
    if not self.use_indices:
      return tensor
    gathered = tf.gather(tensor, self.indices, axis=axis)
    if gathered.dtype != self.dtype:
      gathered = tf.cast(gathered, self.dtype)
    return gathered

  def normalize(self, X, gather_x=True):
    if gather_x:
      X = self.gather(X)
    S = tf.subtract(X, self.lowBound, name="norm_sub")
    return tf.divide(S, self._dividion_denomiator, name="norm_div")
  
  def convert_inputs(self, y_true, y_pred, set_dtype=True):
    y_pred = tf.convert_to_tensor(y_pred)    
    y_true = tf.cast(y_true, y_pred.dtype)
    if set_dtype:
      self.setDtype(y_pred.dtype)
    return y_true, y_pred

  def setDtype(self, dtype):
    self.dtype = dtype

  def get_config(self):
    base_config = super().get_config()
    return {**base_config, "lb":self.lb, "up": self.up, "epsilon": self.epsilon}
  

class NormalizedMeanAbsoluteError(NormalizedBaseClass):
  def __init__(self, lb=[], up=[], indices=[], **kwargs):
    super().__init__(**kwargs)
  
  def call(self, y_true, y_pred):
    y_true, y_pred = self.convert_inputs(y_true, y_pred)
    y_true_norm, y_pred_norm = self.normalize(y_true), self.normalize(y_pred, False)
    return tf.math.reduce_mean(tf.abs(y_pred_norm - y_true_norm), axis=-1)
  def get_config(self):
    base_config = super().get_config()
    return {**base_config, "lb":self.lb, "up": self.up, "epsilon": self.epsilon}

class NormalizedMeanSquaredError(NormalizedBaseClass):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  def call(self, y_true, y_pred):
    y_true, y_pred = self.convert_inputs(y_true, y_pred)
    y_true_norm, y_pred_norm = self.normalize(y_true), self.normalize(y_pred, False)
    return tf.math.reduce_mean(tf.math.square(y_pred_norm - y_true_norm), axis=-1)
  def get_config(self):
    base_config = super().get_config()
    return {**base_config, "lb":self.lb, "up": self.up, "epsilon": self.epsilon}