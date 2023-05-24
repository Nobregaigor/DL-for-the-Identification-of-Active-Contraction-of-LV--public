import tensorflow as tf
import matplotlib 
import matplotlib.pyplot as plt

def scatter_accuracy(ys, preds):
  try:
    if ys.shape != preds.shape:
      ys = tf.reshape(ys, preds.shape)
    e = abs(ys - preds)
    fig, ax = plt.subplots(figsize=(10,10))
    cb = ax.scatter(ys, preds, c=e, cmap="jet")
    ax.grid()
    ax.set_ylabel("Predicted")
    ax.set_xlabel("True")
    ax.set_title("Accuracy scatter plot")
    fig.colorbar(cb, ax=ax)
  except:
    print("Failed generating scatter plot.")