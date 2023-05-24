import numpy as np
from scipy.signal import savgol_filter

class Curve_Manipulator():
  @staticmethod
  def smooth(arr, window=3, polyorder=1, start=None, end=None):
    """
      Uses savgol_filter with given window length and polynomial order to smooth 
      the input array.
    """
    left = 0 if start is None else start
    right = 0 if end is None else end
    if right == 0 and left == 0:
      return savgol_filter(arr, window, polyorder)

    if right == 0:
      arr[left:] = savgol_filter(arr[left:], window, polyorder)
    else:
      arr[left:right] = savgol_filter(arr[left:right], window, polyorder)
    return arr
  @staticmethod
  def smooth2d(arr, window=3, polyorder=1):
    a = Curve_Manipulator.smooth(arr[:, 0], window=window, polyorder=polyorder)
    b = Curve_Manipulator.smooth(arr[:, 1], window=window, polyorder=polyorder)
    x = np.hstack([a.reshape([-1,1]), b.reshape([-1,1])])
    return x
  
  @staticmethod
  def convolve(arr, window=3):
    box = np.ones(window)/window
    return np.convolve(arr, box, mode='same')
  
  @staticmethod
  def convolve2d(arr, window=3):
    a = Curve_Manipulator.convolve(arr[:, 0], window=window)
    b = Curve_Manipulator.convolve(arr[:, 1], window=window)
    x = np.hstack([a.reshape([-1,1]), b.reshape([-1,1])])
    return x

  @staticmethod
  def adjust_ends(self, arr, to_value=0.0, 
      window=2, steps=4, runs=1, increase_window=5, window_step=1):
    """
      Set endpoints of a gamma waveform to a given value. 
      It also performs shallow adjustments to surrounding elements. 
    """
    arr = np.reshape(arr, [-1,])
    arr[0], arr[-1] = to_value, to_value
    count = 0
    for _ in range(runs):
      if increase_window > 0:
        count += 1
        if count == increase_window:
          window += window_step
          count = 0

      # adjust endpoints given window size and steps
      for iright in range(1, (window * steps) + 1, window):
        ileft = iright - 1
        arr[ileft+1:iright+1] = np.full(
            arr[ileft:iright].shape, np.mean(arr[ileft:iright+1]))

      for ileft in range(-2, - ((window * steps) + 1), -window):
        iright = ileft + 1
        elems = [arr[el] for el in range(ileft, iright+1)]
        arr[ileft:iright] = np.full(arr[ileft:iright].shape, np.mean(elems))

      arr[0], arr[-1] = to_value, to_value
    arr = np.reshape(arr, [-1,1])
    return arr
  
  @staticmethod
  def resize_and_interpolate(arr, new_length):
    xs = np.linspace(0.0, 1.0, new_length)
    ys = np.linspace(0.0, 1.0, arr.shape[0])
    return np.interp(xs, ys, arr)
  @staticmethod
  def resize_and_interpolate_2D(arr, new_length):
    a = Curve_Manipulator.resize_and_interpolate(arr[:, 0], new_length)
    b = Curve_Manipulator.resize_and_interpolate(arr[:, 1], new_length)
    x = np.hstack([a.reshape([-1,1]), b.reshape([-1,1])])
    return x
