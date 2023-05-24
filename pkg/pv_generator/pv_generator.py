from pathlib import Path
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy.spatial import ConvexHull
import re
import pandas as pd
import os
import numpy as np

def gen_pv(base_refs,
  # randomize curves
  random_vertical_translation=(-0.15, 0.225),
  random_horizontal_translation=(-0.15, 0.225), 
  random_pressure_scale=(0.3, 1.125),
  random_volume_scale=(0.3, 1.2),
  # mutation (uses either pressure or volume from other pv curve)
  chance_of_mutation=0.05, # minimum value for mutation to occur
  chance_of_pressure_mutation=0.4, # sum of chances must not be greater than one
  chance_of_volume_mutation=0.4,
  # contrains
  set_initial_vol_to=None,
  set_initial_pres_to=None,
  set_pressure_deltas_to=(0.1, 1.7, 0.9), # min, max, factor
  set_volume_deltas_to=(0.3, 0.7, 0.9),   # min, max, factor
  set_pressure_bounds_to=(0.001, None),
  set_volume_bounds_to=(None, 1.0),
  set_max_initial_pressure_to=None,
  allow_initial_values_to_exceed_bounds=True,
  # final modifications
  resize_to=200,
  chance_of_roll=None,
  chance_of_noise=None,
  max_signal_noise=0.0025,
  ):
  
  import tensorflow as tf
  from tensorflow.dtypes import float32, int32
  from tensorflow.random import uniform as tf_ru
  from functools import partial

  def is_valid(value):
    return not value is None

  # choose a reference
  idx_ref = tf_ru([1], minval=0, maxval=len(base_refs), dtype=int32).numpy()[0]
  base = base_refs[idx_ref].copy()

  # get pressure and volume
  ps = base["pressure"]
  vs = base["volume"]

  # === MUTATION ===
  # validate mutation argument
  if is_valid(chance_of_mutation):
    # if mutation chance (random value) is within mutation range (change of mut)
    # we can apply a mutation
    mutation_chance = tf_ru([1], minval=0.0, maxval=1.0, dtype=float32).numpy()[0]
    if mutation_chance < chance_of_mutation:
      # get another reference idx (might be same idx) and base
      mut_idx_ref = tf_ru([1], minval=0, maxval=len(base_refs), dtype=int32).numpy()[0]
      mut_base = base_refs[mut_idx_ref].copy()
      # get random value to apply mutation on axis
      where = tf_ru([1], minval=0.0, maxval=1.0, dtype=float32).numpy()[0]
      if where <= chance_of_pressure_mutation:
        ps = mut_base["pressure"]
      if where >= 1.0 - chance_of_volume_mutation:
        vs = mut_base["volume"]

  # === RANDOM MODIFICATIONS ===
  # --- apply random translations ---
  # vertical translations
  if is_valid(random_vertical_translation):
    vmin, vmax = random_vertical_translation
    vertical_shift = tf_ru([1], minval=vmin, maxval=vmax, dtype=float32).numpy()
    ps += vertical_shift
  # horizontal translation
  if is_valid(random_horizontal_translation):
    vmin, vmax = random_horizontal_translation
    min_v_existing = np.min(vs)
    max_v_existing = np.max(vs)
    ef = (max_v_existing - min_v_existing) / max_v_existing
    if ef > 0.7: # too high -> need to lower
      vmin = 0.0
      vmax = vmax if vmax > 0.0 else 0.3 
    if ef < 0.3: # too low -> need to increase
      vmin = vmin if vmin < 0.0 else -0.3
      vmax = 0.0
    horizontal_shift = tf_ru([1], minval=vmin, maxval=vmax, dtype=float32).numpy()
    vs += horizontal_shift
  
  # --- apply random scale ---
  # scale pressure
  if is_valid(random_pressure_scale):
    vmin, vmax = random_pressure_scale
    ps_scale = tf_ru([1], minval=vmin, maxval=vmax, dtype=float32).numpy()
    ps *= ps_scale
  # scale volume
  if is_valid(random_volume_scale):
    vmin, vmax = random_volume_scale
    vs_scale = tf_ru([1], minval=vmin, maxval=vmax, dtype=float32).numpy()
    vs *= vs_scale
  
  # === CONTRAINS ===
  # --- apply delta contrains ---
  # set constrain on pressure deltas
  if is_valid(set_pressure_deltas_to):
    vmin, vmax, df = set_pressure_deltas_to
    if not is_valid(vmin):
      vmin = ps.min()
    if not is_valid(vmax):
      vmax = ps.max()
    delta_p = abs(ps.max() - ps.min())
    while (delta_p < vmin) or (delta_p > vmax):
      if delta_p < vmin:
        ps *= abs(1.0 + delta_p * df)
      if delta_p > vmax:
        ps *= abs(1.0 - delta_p * df)
      delta_p = abs(ps.max() - ps.min())
  # set constrain on volume deltas
  if is_valid(set_volume_deltas_to):
    vmin, vmax, df = set_volume_deltas_to
    if not is_valid(vmin):
      vmin = vs.min()
    if not is_valid(vmax):
      vmax = vs.max()
    delta_v = abs(vs.max() - vs.min())
    while (delta_v < vmin) or (delta_v > vmax):
      if delta_v < vmin:
        vs *= abs(1.0 + delta_v * df)
      if delta_v > vmax:
        vs *= abs(1.0 - delta_v * df)
      delta_v = abs(vs.max() - vs.min())
  # --- apply max initial pressure ---
  if is_valid(set_max_initial_pressure_to):
    if ps[0] > set_max_initial_pressure_to:
      ps += set_max_initial_pressure_to - ps[0]
      
  # --- apply fixed initial params ---
  # fix initial volume
  if is_valid(set_initial_vol_to):
    if vs[0] != set_initial_vol_to:
      vs += set_initial_vol_to - vs[0]
  # fix initial volume
  if is_valid(set_initial_pres_to):
    if ps[0] != set_initial_pres_to:
      ps += set_initial_pres_to - ps[0]
  # --- apply boundaries ---
  # fix pressure bounds
  if is_valid(set_pressure_bounds_to):
    minp, maxp = set_pressure_bounds_to
    if is_valid(minp):
      ps = np.where(ps < minp, minp, ps)
    if is_valid(maxp):
      ps = np.where(ps > maxp, maxp, ps)
  # fix volume bounds
  if is_valid(set_volume_bounds_to):
    minv, maxv = set_volume_bounds_to
    if is_valid(minv):
      vs = np.where(vs < minv, minv, vs)
    if is_valid(maxv):
      vs = np.where(vs > maxv, maxv, vs)
  # update initial values (if requested)
  if allow_initial_values_to_exceed_bounds:
    if is_valid(set_initial_vol_to):
      vs[0] = vs[-1] = set_initial_vol_to
    if is_valid(set_initial_pres_to):
      ps[0] = ps[-1] = set_initial_pres_to

  # roll values to match max vol and min pres
  max_vol = vs.max() * 0.98
  mask = vs >= max_vol
 
  min_pressure = ps[mask].min()
  rollby = np.where(ps == min_pressure)[0][-1] + 1
  # print(min_pressure, rollby)
  ps = np.roll(ps, -rollby)
  vs = np.roll(vs, -rollby)

  # === ROLL ===
  # --- apply roll ---
  if is_valid(chance_of_roll):
    roll_chance = tf_ru([1], minval=0.0, maxval=1.0, dtype=float32).numpy()[0]
    if roll_chance < chance_of_roll:
      roll_by = tf_ru([1], minval=0, maxval=ps.shape[0], dtype=int32).numpy()
      ps = np.roll(ps, roll_by)
      vs = np.roll(vs, roll_by)

  # === SHAPING FOR RETURN ===
  # --- apply resize ---
  if is_valid(resize_to):
    xs = np.linspace(0.0, 1.0, resize_to)
    ts = np.linspace(0.0, 1.0, ps.shape[0])
    ps = np.interp(xs, ts, ps)
    vs = np.interp(xs, ts, vs)
  
  # reshape and tensorfy
  ps = ps.reshape([-1, 1])
  vs = vs.reshape([-1, 1])
  pv = tf.concat([ps, vs], -1)
  pv = tf.cast(pv, float32)

  # === NOISE ===
  # --- apply noise ---
  if is_valid(chance_of_noise):
    noise_chance = tf_ru([1], minval=0.0, maxval=1.0, dtype=float32).numpy()[0]
    if noise_chance < chance_of_noise:
      signal_noise_seed = tf.random.uniform([2], 0, max_signal_noise)
      signal_noise = tf.random.normal(pv.shape, signal_noise_seed[0], signal_noise_seed[1])
      pv = tf.add(pv, signal_noise)
  
  return pv

def gen_valid_pv(base_refs, min_ef=0.3, max_ef=0.7, min_max_ps=0.6, max_max_ps=1.25, **kwargs):
  from collections import deque
  
  valid = False
  while valid == False:
    pv = gen_pv(base_refs, **kwargs)

    checks = deque()

    checks.append(np.var(pv[:, 0]) >= 0.005)
    checks.append(np.var(pv[:, 1]) >= 0.005)

    # min dims
    checks.append(abs(np.max(pv[:, 1]) - np.min(pv[:, 1])) >= 0.1) 
    checks.append(abs(np.max(pv[:, 0]) - np.min(pv[:, 0])) >= 0.1) 

    # bounds
    checks.append(np.min(pv[:, 0]) >= 0)
    checks.append(np.min(pv[:, 1]) >= 0)

    vs = pv[:, 1]
    min_v_existing = np.min(vs)
    max_v_existing = np.max(vs)
    ef = (max_v_existing - min_v_existing) / max_v_existing
    checks.append(ef <= max_ef)
    checks.append(ef >= min_ef)

    ps = pv[:, 0]
    checks.append(np.max(ps) >= min_max_ps)
    checks.append(np.max(ps) <= max_max_ps)

    if all(checks) == True:
      valid = True
    
  return pv

def visualize_pv_gen(pv_gen_fun, pv_samples, n_cols=4, n_pv_per_col=10):
  fig, allaxs = plt.subplots(n_cols, 3, figsize=(28,7*n_cols))
  for i in range(n_cols):
    axs=allaxs[i]
    for _ in range(n_pv_per_col):
      pv = pv_gen_fun(pv_samples)

      ps = pv[:, 0]
      vs = pv[:, 1]
      xs = np.linspace(0,1,ps.shape[0])

      axs[0].plot(vs, ps)
      axs[0].axis('equal')
    
      axs[1].plot(xs, ps, linestyle="-")
      axs[2].plot(xs, vs, linestyle="--")

      axs[1].scatter(xs, ps, marker=".", s=20)
      axs[2].scatter(xs, vs, marker=".", s=20)

      axs[0].set_xlabel("Volume")
      axs[0].set_ylabel("Pressure")
      axs[0].set_title("PV Loop")
      axs[0].grid(linestyle='-', linewidth=0.5)

      axs[1].set_xlabel("xs")
      axs[1].set_ylabel("Pressure")
      axs[1].set_title("Pressure Waveform")
      axs[1].grid(linestyle='-', linewidth=0.5)

      axs[2].set_xlabel("xs")
      axs[2].set_ylabel("Volume")
      axs[2].set_title("Volume Waveform")
      axs[2].grid(linestyle='-', linewidth=0.5)