import json
import pandas as pd
import numpy as np
import tensorflow as tf

def get_indexes(header, of_keys):
  return [header[key] for key in of_keys]

def csv_ds_reader(filepaths, 
                  header,
                  label_keys,
                  feature_keys,
                  int_labels=False,
                  reshape_to=None,
                  batch_size=32,
    repeat=1, n_readers=5, 
    shuffle_buffer_size=10000, 
    cache=True, 
    n_read_threads=None, 
    n_parse_threads=5):
  
  label_indexes = get_indexes(header, label_keys)
  feature_indexes = get_indexes(header, feature_keys)
  
  def get_features_labels(line):
    """
      Re-structures DS to read csv field and return 'features, labels'
    """
    defs = [0.] * (len(header) + 1) #n_labels
    fields = tf.io.decode_csv(line, record_defaults=defs)
 
    # cm-lookup -> CM1 -> (CMs), (FB, PV)
    _labels  = tf.stack([fields[i] for i in label_indexes])
    features = tf.stack([fields[i] for i in feature_indexes])
   
    try:
      tf.debugging.check_numerics(_labels, message='Checking _labels')
    except Exception as e:
      assert "Checking labels: Tensor had NaN values" in e.message
    
    try:
      tf.debugging.check_numerics(features, message='Checking features')
    except Exception as e:
      assert "Checking features: Tensor had NaN values" in e.message

    if int_labels is True:
      _labels = tf.cast(_labels, tf.int32, "label_int")

    if reshape_to is not None:
      _labels = tf.reshape(_labels, reshape_to)
  
    return features, _labels
  
  # get files
  ds = tf.data.Dataset.list_files(str(pathlib.Path(filepaths)/'*.csv'))
  # read in random order cycling through files
  ds = ds.interleave(
      lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
      cycle_length=n_readers, num_parallel_calls=n_read_threads
  )
  # shuffle and repeat
  ds = ds.shuffle(shuffle_buffer_size).repeat(repeat)
  # preprocess dataset (must provide a function)
  ds = ds.map(get_features_labels, num_parallel_calls=n_parse_threads)
  # cache dataset
  ds = ds.cache()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(1)
  return ds