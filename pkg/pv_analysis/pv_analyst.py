from .modules import Curve_Manipulator
from .modules.logger import LoggerWrapper
# from .modules import LoggerWrapper
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.cm as plt_cm

class PV_Analyst(Curve_Manipulator, LoggerWrapper):
  def __init__(self, pv=None, logName="PVA", 
               pv_phases_params=None,
               **kwargs):
    super().__init__(logName=logName, **kwargs)
    self._pv = None
    self._ps = None
    self._vs = None
    self._ts = None
    # regions determine the 'four corners' of the PV loop
    self._regions = None
    self._r_brkpts = None
    # phases set the four major parts of the cardiac cycle
    self._phases = None
    self._p_brkpts = None

    if not pv is None:
      self._update_pv(pv)

    self._phases_names = {
      0: "Isovolumetric Contraction",
      1: "Ejection",
      2: "Isovolumetric Relaxation",
      3: "Centricles Filling",
    }
    self._phases_dict = {
      "IC": 0,
      "EJ": 1,
      "IR": 2,
      "CF": 3
    }

    self._ref_pvl = None

    # function_params
    self._param_compute_pv_phases = {
      "alpha":30, # angle of weighted p,v for first pass
      "beta":0.25, # percentage of maximum gradiant value to be used in threshold
      "ic_lambda": 0.42, # threshold for percentage of values above beta*max_grad for ic
      "ir_lambda": 0.42, # threshold for percentage of values above beta*max_grad for ir
      "cf_delta":0.10, # min delta for cf gradients be considered linear
      "ic_gamma":0.9875, 
      "ir_gamma":1.04,
    }
    if not pv_phases_params is None:
      self.set_phases_params(pv_phases_params)

    # scale factors
    self._max_pressure = 16.0
    self._initial_volume = 95.212664 # 122775.39 * 0.001 # mm3 to ml

    self.dlog("PVA created.")
  def _update_pv(self, pv):
    self._pv = pv
    self._ps = pv[:, 0]
    self._vs = pv[:, 1]
  def _reset_pv(self):
    self._pv = None
    self._ps = None
    self._vs = None
    self._regions = None
    self._r_brkpts = None
    self._phases = None
    self._p_brkpts = None
    self._ts = None
  def update(self, pv):
    self._update_pv(pv)
  def reset(self):
    self.dlog("Reseting PV.")
    self._reset_pv()
  def set_phases_params(self, params):
    if isinstance(params, dict):
      self._param_compute_pv_phases = params
    else:
      self.elog("Could not set params. Params must be a dictionary.")
  def get_pv(self):
    if self._pv is None:
      self.wlog("pv was not found, maybe it wasn't initialized.")
    return self._pv
  def get_ts(self):
    if self._ts is None:
      self.wlog("ts was not found, maybe it wasn't computed.\
                use 'distribute_timespace()' to compute it.")
    return self._ts
  def get_pv_regions(self, pv=None):
    if pv is None:
      pv = self.get_pv()
    if self._regions is None:
      self.dlog("Regions were not previously computed. Computing them now.")
      self.compute_pv_regions(pv)
    return self._regions, self._r_brkpts
  def get_pv_phases(self, pv=None):
    if pv is None:
      pv = self.get_pv()
    if self._phases is None:
      self.dlog("Phases were not previously computed. Computing them now.")
      self.compute_pv_phases(pv, **self._param_compute_pv_phases)
    return self._phases, self._p_brkpts
  def get_pv_phase_index(self, p):
      """
        Returns the index of a phase breakpoint
        p is an int or a string determining the phase region
        If phases is not computed, it returns None
      """
      if self._p_brkpts is None:
        self.dlog("phases breakpoints were not computed.")
        return None
      if isinstance(p, int):
        return self._p_brkpts[p]
      else:
        return self._p_brkpts[self._phases_dict[p]]
  def get_pv_phase_value(self, p):
    idx = self.get_pv_phase_index(p)
    if idx is None:
      self.dlog("Could not get phase value. \
        PV phases might not have been computed.")
    return self._pv[idx]
  def get_ref_pvl(self, pv=None, window=3, return_index=False):
    """
      Returns the value of (p,v) for lowest volume in PV loop, assuming it
      happens at Isovolumetric relaxation
    """
    if pv is None:
      pv = self._pv
    
    # get isovolumetric relaxation region
    _, p_brkpts = self.get_pv_phases(pv)
    region = pv[p_brkpts[2]:p_brkpts[3]]
    # find minimum volume index
    focal_idx = np.argmin(region[:, 1]) + p_brkpts[2]
    if return_index:
      return focal_idx
    # set window
    if window > 0:
      left, right = focal_idx - window, focal_idx + window
      return np.mean(pv[left:right], axis=0)
    else:
      return pv[focal_idx]
  def discretize_region(self, pv_pt, p_range, v_range):
    """
      This function computes four regions of the PV curve, based on the 
      distance from four corners of a square. It can also be though as
      'partitioning' sub-function.
      
      It requires a PV point (pressure, volume) and the range of P and V,
      in terms of max and min of thesevalues.

      It returns the region of the point, an int from 0 to 3
    """
    def dist(a,b):
      return abs(a - b)
    def closest(x, q):
      """Returns the index in which x is closest to in a q array/tuple"""
      qs = [dist(x, v) for v in q]
      return np.argmin(qs)

    # find indexes of closes values
    p_idx = closest(pv_pt[0], p_range)
    v_idx = closest(pv_pt[1], v_range)
    
    # Detemine the region in which the point is located at based on the 
    # four corners of a square
    # IC: max vol and min pres
    if v_idx == 1 and p_idx == 0:
      return 0 
    # EJ: max vol and max pres
    if v_idx == 1 and p_idx == 1:
      return 1
    # IR: min vol and max pres
    if v_idx == 0 and p_idx == 1:
      return 2
    # CF: min vol and min pres
    if v_idx == 0 and p_idx == 0:
      return 3 
  def compute_pv_regions(self, pv=None, 
                   discrete_ends=False, 
                   discrete_value=4):
    """
      Using 'self.discretize_region' as a method of determining the region of a 
      given pv point, this function iterates through the PV Loop and computes
      the region of all points in the curve.

      By default, it returns an array of same length as PV with integers 
      discretisizing the region of each point.

      If discrete_ends is true, it distinguishes the last part of the curve,
      which belongs to region 0 but it happens at the end of the loop. The
      value of 0 at end will be replaced with the provided discrete_value.
    """
    # ---
    # Update local PV
    if not pv is None:
      self._update_pv(pv)
    else:
      pv = self._pv

    # --- 
    # get ranges of min_max of pressure and vl
    min_max_pr = [np.min(self._ps), np.max(self._ps)]
    min_max_vl = [np.min(self._vs), np.max(self._vs)]
    
    # ---
    # set regions of each point based on 'self.discretize_region'
    regions = np.zeros([pv.shape[0],], dtype=np.int16)
    for i, pt in enumerate(pv):
      # value of region from 0 to 3
      rval = self.discretize_region(pt, min_max_pr, min_max_vl)
      # if end values must be separated from beginning values,
      # replace them with the given replacement value
      if discrete_ends and i > 0:
        if regions[i-1] > 0 and rval == 0:
          rval = discrete_value
      regions[i] = rval

    # compute break_points
    breakpoints = np.nonzero(np.gradient(regions))[0][::2] + 1
    # update object state
    self._regions = regions
    self._r_brkpts = breakpoints
    
    return regions, breakpoints
  def compute_pv_phases(self, pv=None,
      alpha=30, beta=0.22, ic_gamma=0.985, ir_gamma=1.05, cf_delta=0.1,
      ic_lambda=0.5, ir_lambda=0.5):
    
    def masked(pv, regions, value):
      return pv[regions == value]
    
    def alpha_weighted_region(region, alpha, pm=1.0, vm=1.0):
      p_weight = np.full([region.shape[0], 1], np.sin(np.radians(alpha))) * pm
      v_weight = np.full([region.shape[0], 1], np.cos(np.radians(alpha))) * vm
      return region * np.hstack([p_weight, v_weight])
    
    # --- 
    # update local pv
    if not pv is None:
      self._update_pv(pv)
    else:
      pv = self._pv

    # ---
    # get regions
    (regions, r_brkpts) = self.get_pv_regions()
    if regions is None:
      (regions, r_brkpts) = self.compute_pv_regions(pv, discrete_ends=True)

    # =========================
    # FIRST PASS -> Compute all four major points
    ic_is_regular = True
    # ----
    # IC: max vol and min pres
    ic_region = masked(pv, regions, 0)
    ic_region = alpha_weighted_region(ic_region, alpha, vm=-1.0)
    ic_idx = np.argmin(np.sum(ic_region, axis=1))
    if ic_idx > r_brkpts[0]:
      ic_idx = len(pv) - (len(ic_region) - ic_idx)
      ic_is_regular = False

    # EJ: max vol and max pres
    ej_region = masked(pv, regions, 1)
    ej_region = alpha_weighted_region(ej_region, alpha)
    ej_idx = np.argmax(np.sum(ej_region, axis=1)) + r_brkpts[0]

    # IR: min vol and max pres
    ir_region = masked(pv, regions, 2)
    ir_region = alpha_weighted_region(ir_region, alpha, vm=-1.0)
    ir_idx = np.argmax(np.sum(ir_region, axis=1)) + r_brkpts[1]

    # CF: min vol and min pres 
    # -- Here we must account for regions where volume is zero. np.argmin 
    #     will return the first minmum value, but it might not be true.
    cf_region = masked(pv, regions, 3)
    cf_region = alpha_weighted_region(cf_region, alpha)
    cf_local_idx = np.argmin(np.sum(cf_region, axis=1))
    cf_vgrads = np.gradient(cf_region[:, 1][cf_local_idx:cf_local_idx+20])
    cf_vgrads = np.where(cf_vgrads < 0.0, 0.0, cf_vgrads)
    cf_vgrads /= np.max(cf_vgrads)
    cf_local_idx += np.argmax(cf_vgrads > cf_delta)
    cf_idx = cf_local_idx + r_brkpts[2]

    # cf_idx = np.argmin(np.sum(cf_region, axis=1)) + r_brkpts[2]


    # =============================
    # SECOND PASS
    def has_isovolumetric_gradients(phase, iso_thresh):
      v_grads = abs(np.gradient(phase[:, 1]))
      max_grad = np.max(v_grads) * beta
      n_bellow_max = len(v_grads[v_grads <= max_grad]) 
      percentage_pts_bellow_max = n_bellow_max/len(v_grads)
      return percentage_pts_bellow_max >= iso_thresh

    # ---
    # Check between ic and ej
    if not ic_is_regular:
      seq_1 = np.vstack([pv[ic_idx:], pv[0:ej_idx]])
      adjusted_initial_p = len(pv) - ic_idx
    else:
      seq_1 = pv[ic_idx:ej_idx]
      adjusted_initial_p = ic_idx
    if has_isovolumetric_gradients(seq_1, ic_lambda):
      self.dlog("re-checking for ic")
      _mask = seq_1[:, 1] >= np.max(seq_1[:, 1]) * ic_gamma
      max_pressure = np.max(seq_1[:, 0][_mask]) 
      ej_idx = np.where(seq_1[:, 0] == max_pressure)[0][0] + adjusted_initial_p
      
    # ---
    # check between ic and ej
    seq_2 = pv[ir_idx:cf_idx]
    if has_isovolumetric_gradients(seq_2, ir_lambda):
      self.dlog("re-checking for ir")
      _mask = seq_2[:, 1] <= np.min(seq_2[:, 1]) * ir_gamma 
      max_pressure = np.max(seq_2[:, 0][_mask]) 
      ir_idx = np.where(seq_2[:, 0] == max_pressure)[0][0] + ir_idx

    # ============================
    # Build regions

    p_brkpts = np.array([ic_idx, ej_idx, ir_idx, cf_idx])
    phases = np.zeros([pv.shape[0],], dtype=np.int16)
    if ic_is_regular:
      phases[ic_idx:ej_idx] = 0
      if ic_idx > 0:
        phases[:ic_idx] = 3
      phases[cf_idx:] = 3
    else:
      phases[ic_idx:] = 0
      phases[0:ej_idx] = 0
      phases[cf_idx:ic_idx] = 3
    
    phases[ej_idx:ir_idx] = 1
    phases[ir_idx:cf_idx] = 2
    # ===========================
    # Log and Update state
    self.dlog("p_brkpts: {}".format(p_brkpts))
    self.dlog("phases: {}".format(phases))
    self._phases = phases
    self._p_brkpts = p_brkpts

    return phases, p_brkpts 
  def plot_pv(self, pv=None, 
    color_mask=None, cbar=False, brkpts=True,
    waveform=False, xs=None, expanded_plot=False, 
    pvl=False, pvl_window=None, legend=True,
    ax=None, figsize=(10,10), title=None,
    ps_color="darkgreen", vs_color="darkgoldenrod",
    no_scatter=False, equal=True,
    ):

    # ---
    if expanded_plot:
      fig = None
      if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
      self.plot_pv(pv=pv, color_mask=color_mask, title=title,
                   cbar=cbar, brkpts=brkpts, pvl=pvl, legend=legend,
                   ps_color=ps_color, vs_color=vs_color, equal=equal,
                    expanded_plot=False, ax=ax[0], waveform=False)
      self.plot_pv(pv=pv, color_mask=color_mask, legend=legend, title=title,
                   cbar=cbar, brkpts=brkpts, xs=xs, pvl=pvl,
                   ps_color=ps_color, vs_color=vs_color, equal=equal,
                    expanded_plot=False, ax=ax[1], waveform=True)
      return fig, ax

    # ---
    fig = None
    if ax is None:
      fig, ax = plt.subplots(1,1, figsize=figsize)
    
    if not waveform and equal:
      ax.axis('equal')
    # else:
    #   ax.set_aspect('auto', adjustable='box')

    # update pv
    if pv is None:
      pv = self.get_pv()
    # elif self._pv is None:
    #   self._update_pv(pv)

    # ---
    # get color maps
    if color_mask == "regions":
      color_map, breakpoints = self.get_pv_regions(pv)
      seq_label = lambda x: "Region {}".format(x)
    elif color_mask == "phases":
      color_map, breakpoints = self.get_pv_phases(pv)
      seq_label = lambda x: self._phases_names[x]
    else:
      color_map, _ = None, None

    # ---
    # get ps, pv
    if pv is None:
      ps = self._ps
      vs = self._vs
    else:
      ps = pv[:, 0]
      vs = pv[:, 1]

    # setup xs if waveform is requested
    xlabel="Volume"
    ylabel="Pressure"
    if waveform:
      twinx = ax.twinx()
      twinx.set_ylabel("Volume")
      xlabel = "xs"
      if xs is None:
        xs = np.linspace(0,1, len(ps))

    # ---
    # Plot PV
  
    if not color_mask is None:
      ps_color = ps_color
      vs_color = vs_color

      # plot with color mask
      if waveform:
        twinx.plot(xs, vs, linewidth=1.0, c=vs_color)
        ax.plot(xs, ps, linewidth=1.0, c=ps_color)
        im = twinx.scatter(xs, vs, c=color_map, marker='.', cmap='rainbow')
        _  = ax.scatter(xs, ps, c=color_map, marker='.', cmap='rainbow')
      else:
        ax.plot(vs, ps)
        im = ax.scatter(vs, ps, c=color_map, cmap='rainbow')
      # show colorbar
      if cbar:
        fig.colorbar(im, ax=ax)
      # plot breakpoints
      if brkpts:
        colors = plt_cm.rainbow(np.linspace(0, 1, len(breakpoints)))
        for i, pt in enumerate(breakpoints):
          if waveform:
            twinx.scatter(xs[pt], vs[pt], c=[colors[i]], marker="X", 
                    s=100, label=seq_label(i))
            ax.scatter(xs[pt], ps[pt], c=[colors[i]], marker="X", s=100)
          else:
            ax.scatter(vs[pt], ps[pt], c=[colors[i]], marker="X", 
                      s=100, label=seq_label(i))
    else:
      # ps_color = "limegreen"
      # vs_color = "gold"
      ps_color = ps_color
      vs_color = vs_color
      # plot without mask
      if waveform:
        twinx.plot(xs, vs, c=vs_color, label="volume")
        ax.plot(xs, ps, c=ps_color, label="pressure")
        if not no_scatter:
          twinx.scatter(xs, vs, c=vs_color)
          ax.scatter(xs, ps, c=ps_color)
      else:
        ax.plot(vs, ps)
        if not no_scatter:
          ax.scatter(vs, ps)

    # ---
    # Plot pvl (if requested)
    if pvl is True:
      pvl_idx = self.get_ref_pvl(return_index=True)

      if waveform:
        twinx.scatter(xs[pvl_idx], vs[pvl_idx], 
                      marker='*', s=200, c="orange", label="PVL Ref")
        ax.scatter(xs[pvl_idx], ps[pvl_idx],  
                   marker='*', s=200, c="orange")
      else:
        ax.scatter(vs[pvl_idx], ps[pvl_idx], 
                   marker='*', s=200, c="orange", label="PVL Ref")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legend:
      plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=5)
    
    if not title is None:
      ax.set_title(title)
    return fig, ax
  def distribute_timespace(self, pv=None, 
      max_time=0.8, ic=0.05, ej=0.35, ir=0.07, cf=0.53, 
      beta1=0.5, beta2=0.5, non_linear_fun=None,
      assume_rolled=False, smooth=True, smooth_args={}):
    # ic: Isovolumetric contraction ( 4~8%)
    # ej: Ejection phase: (33%~40%)
    # ir: Isovolumic relaxation: (6%~15%)
    # cf: Centricles filling: (50%~60%)

    def apply_nonlinear_distribution(x, beta):
      if not non_linear_fun is None:
        return non_linear_fun(x, beta=beta)
      else:
        a, b = x[0], x[-1]
        d = b-a
        return (d/(np.exp(beta * d)-1)) * (np.exp(beta * (x-a)) - 1) + a

    # make sure time distribution sums up to 1.0
    t_distr = np.array([ic, ej, ir, cf])
    t_distr = t_distr/t_distr.sum()
    
    # --- 
    # get pv
    if pv is None:
      pv = self.get_pv()
    # ---
    # get phases 
    phases, _ = self.get_pv_phases(pv)

    # --- 
    # we need pv to start at IC for this algorithm
    if not assume_rolled:
      self.dlog("PV was not assumed to be rolled -> Rolling pv by IC.")
      self.roll_pv_by_phase("IC")
      phases, _ = self.get_pv_phases(pv)
    else:
      self.dlog("PV assumed to be rolled. Make sure it starts with IC \
                breakpoint, otherwise algorithm will not work properly.")
    # ---
    # set variables
    ts = []
    prv_ts = 0.0
    # start algorithm
    for i, scale in enumerate(t_distr):
      # mask pv and get only points on current phase
      n_data_in_phase = len(pv[phases == i])

      # compute next timestep breakpoint
      next_ts = max_time * scale + prv_ts

      # (a) create linear time distribution
      if i != 3:
        curr_ts = np.linspace(prv_ts, next_ts, n_data_in_phase+1) #linear
      else:
        curr_ts = np.linspace(prv_ts, next_ts, n_data_in_phase) #linear

      # (b) Adjust for non-linear distribution
      if i == 1: # ej
        curr_ts = apply_nonlinear_distribution(curr_ts, beta1)
      if i == 3: # cf
        curr_ts = apply_nonlinear_distribution(curr_ts, beta2)
      
      # (c) record values to array
      if i != 3:
        ts.append(curr_ts[:-1]) # remove extra pt to match len and no overlap
      else:
        ts.append(curr_ts)

      # set next ts breakpoint
      prv_ts = next_ts

    # convert values to numpy and stack them in correctly order
    ts = np.hstack(ts)
    if smooth:
      ts = self.smooth(ts, **smooth_args)

    # save ts
    self._ts = ts

    if not assume_rolled:
      return ts, pv
    else:
      return ts,
  def set_max_pressure(self, max_pressure):
    self._max_pressure = max_pressure
  def set_initial_volume(self, initial_volume):
    self._initial_volume = initial_volume
  def scale_pv(self, max_pressure=-1, initial_volume=-1, self_scale=True):
    if max_pressure == -1:
      max_pressure = self._max_pressure
    if initial_volume == -1:
      initial_volume = self._initial_volume
    ps_scales = self.scale_pressure(self._ps, max_pressure)
    ps_scales = np.reshape(ps_scales, [-1, 1])
    vs_scales = self.scale_volume(self._vs, initial_volume)
    vs_scales = np.reshape(vs_scales, [-1, 1])
    pv_new = np.concatenate([ps_scales, vs_scales], -1)
    if self_scale:
      self._reset_pv()
      self._update_pv(pv_new)
    return pv_new
  def scale_pressure(self, tensor, max_pressure=-1):
    if max_pressure == -1:
      max_pressure = self._max_pressure
    tensor = np.where(tensor < 0.0, 0.0, tensor)
    return tensor * max_pressure
  def scale_volume(self, tensor, initial_volume=-1): 
    if initial_volume == -1:
      initial_volume = self._initial_volume
    return tensor * initial_volume
  def roll_pv_by_phase(self, p="IC"):
    """
      Roll pv curve to match given p phase as first element.
      Default p is Isovolumetric Contraction
    """
    idx = self.get_pv_phase_index(p)
    # check valid index
    if idx is None:
      self.wlog("Could not row PV. PV phases might not have been computed.")
      return None

    self.dlog("Phase: {}, idx: {} -> rollby: {}".format(p, 
                                                        idx, 
                                                        -idx))
    
    def roll_if_valid(value, rollby=idx):
      if not value is None:
        if idx == 0:
          return value
        else:
          return np.roll(value, -rollby, axis=-1)
      else:
        return None

    self._pv = roll_if_valid(self._pv)
    self._ps = roll_if_valid(self._ps)
    self._vs = roll_if_valid(self._vs)
    self._regions = roll_if_valid(self._regions)
    self._phases = roll_if_valid(self._phases)
    if not self._r_brkpts is None:
      self._r_brkpts -= idx
    if not self._p_brkpts is None:
      self._p_brkpts -= idx