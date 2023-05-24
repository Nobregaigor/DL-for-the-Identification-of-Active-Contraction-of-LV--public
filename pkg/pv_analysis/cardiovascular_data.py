from scipy.spatial import ConvexHull # required for area computation
import numpy as np
from . import PV_Analyst
import pandas as pd

class CardiovascularData(PV_Analyst):
  def __init__(self, pv, 
               HR=-1, 
               scale=False, 
               max_pressure=-1, initial_volume=-1, 
               logName="CD",
               **kwargs):
    super().__init__(pv=pv, logName=logName, **kwargs)
    self._update_pv(pv)

    # set scaling parameters
    if max_pressure != -1:
      self.set_max_pressure(max_pressure)
    if initial_volume != -1:
      self.set_initial_volume(initial_volume)
    # scale pv, if requested
    if scale == True:
      self.scale_pv()

    # make sure phases are computed
    if self._phases is None:
      self.compute_pv_phases(pv)

    # --- Input parameters
    # heart rate
    self.HR = HR 

    # --- Mins and Maxs
    self.Vmin = self.get_min_volume()
    self.Vmax = self.get_max_volume()
    self.Pmin = self.get_min_pressure()
    self.Pmax = self.get_max_pressure()
    # --- Diastole
    # -- start of diastole
    self.start_diastole = self.get_start_diastole()
    self.SDP = self.get_start_diastole_pressure()
    self.SDV = self.get_start_diastole_volume()
    # ---
    # -- End diatole values
    self.end_diastole = self.get_end_diastole()
    self.EDV = self.get_end_diastole_volume()
    self.EDP = self.get_end_diastole_pressure()
    # --- Systole
    # -- Start systole values
    self.start_systole = self.get_start_systole()
    self.SSV = self.get_start_systole_volume()
    self.SSP = self.get_start_systole_pressure()
    # -- End Systole values
    self.end_systole = self.get_end_systole()
    self.ESV = self.get_end_systole_volume()
    self.ESP = self.get_end_systole_pressure()
  
    # ---
    # stroke volume
    self.SV = self.comp_stroke_volume()
    # stroke work
    self.SW = self.comp_stroke_work()
    # cardiac output
    self.CO = self.comp_cardiac_output()
    # coupling ratio
    self.CR = self.comp_coupling_ratio()
    # arterial elastance
    self.Ea = self.comp_arterial_elastance()
    # end-systolic elastance
    self.Ees = self.comp_end_systolic_elastance()
    # Ejaction fraction
    self.EF = self.comp_ejaction_fraction()

    # ---
    # Values with different names (from other references)
    # diastolic blood pressure -> pressure onset of ejection
    self.DBP = self.SDP
    # systolic blood pressure -> maximum pressure
    self.SBP = self.Pmax
    # End-systolic pressure -> pressure at end of systole
    self.Pes = self.ESP
    # Pressure existing in the left atrium
    self.LAP = self.SSP

  # ============================
  # GET METHODS
  # These methods do not require computations
  # e.g. do not rely on equations
  
  # ---
  # Mins and Max
  def get_max_volume(self):
    return np.max(self._vs)
  def get_min_volume(self):
    return np.min(self._vs)
  def get_max_pressure(self):
    return np.max(self._ps)
  def get_min_pressure(self):
    return np.min(self._ps)
  # --- Diastole
  # Start diastole
  def get_start_diastole(self):
    return self.get_pv_phase_value("EJ")
  def get_start_diastole_volume(self):
    return self.get_start_diastole()[1]
  def get_start_diastole_pressure(self):
    return self.get_start_diastole()[0]
  # End diastole
  def get_end_diastole(self):
    return self.get_pv_phase_value("IC")
  def get_end_diastole_volume(self):
    return self.get_end_diastole()[1]
  def get_end_diastole_pressure(self):
    return self.get_end_diastole()[0]
  # --- Systole
  # Start systole
  def get_start_systole(self):
    return self.get_pv_phase_value("CF")
  def get_start_systole_volume(self):
    return self.get_start_systole()[1]
  def get_start_systole_pressure(self):
    return self.get_start_systole()[0]
  # End systole
  def get_end_systole(self):
    return self.get_pv_phase_value("IR")
  def get_end_systole_volume(self):
    return self.get_end_systole()[1]
  def get_end_systole_pressure(self):
    return self.get_end_systole()[0]
  
  # ===================
  # COMP methods
  # These are equation-based methods

  # stroke volume
  def comp_stroke_volume(self):
    """
      Stroke volume is the volume of blood ejected by a
      ventricle in a single contraction. It is the difference
      between the end diastolic volume (EDV) and the end
      systolic volume (ESV).
    """
    return self.EDV - self.ESV
  # stroke work
  def comp_stroke_work(self):
    """
      Defined as the area enclosed by the PV loop
    """
    hull = ConvexHull(self._pv)
    return hull.area
  # cardiac output
  def comp_cardiac_output(self, HR=-1):
    """
      Defined as the amount of blood pumped by the 
      ventricle in unit time. 
      HR is the number of heart beats per minute
    """
    if HR == -1: # no HR was provided. Look for self.HR
      if self.HR == -1:
        self.dlog("No HR was provided. CO could not be computed.")
        return -1
      else:
        HR = self.HR
    
    CO = self.SV * HR
    self.CO = CO
    return CO
  # coupling ratio
  def comp_coupling_ratio(self):
    """
      Defined as the indication of transfer of power from the ventricle
      to the peripheral vasculature
    """
    return self.SV/self.ESV
  # Arterial Elastance
  def comp_arterial_elastance(self):
    """
      Defined as the measurement of arterial load and its impact on the
      ventricle. Calculated as the simple ratio of ventricular
      end-systolic pressure to stroke volume.
    """
    return self.ESP/self.SV
  # End-systolic elastance
  def comp_end_systolic_elastance(self):
    """
      Defined as the slope at the end systolic PV
    """
    return self.ESP/self.ESV 
  # ejaction fraction
  def comp_ejaction_fraction(self):
    """
      Ejection fraction is the ratio of the volume of blood
      ejected from the ventricle per beat (stroke volume)
      to the volume of blood in that ventricle at the end
      of diastole. It is widely clinically misunderstood as
      an index of contractility, but it is a load dependent
      parameter. Healthy ventricles typically have ejection
      fractions greater than 55%.
    """
    return (self.SV/self.EDV) * 100.0
  # pressure-volume area --> NEED IMPLEMENTATION
  def comp_pressure_volume_area(self):
    """
      The PVA represents the total mechanical energy (TME)
      generated by ventricular contraction. This is equal to
      the sum of the stroke work (SW), encompassed within
      the PV loop, and the elastic potential energy (PE).
    """
    return -1
    # return self.PE + self.SW
  
  # ================
  # Display methods
  def print(self, 
              minmax=True, diastole=True, 
              systole=True, computed=True):
    # --- Mins and Maxs
    if minmax:
      print("="*50)
      print("Min-Max values:")
      print("-"*50)
      print("Vmin: {}".format(self.Vmin))
      print("Vmax: {}".format(self.Vmax))
      print("Pmin: {}".format(self.Pmin))
      print("Pmax: {}".format(self.Pmax))
    if diastole:
      print("="*50)
      print("Diastole values:")
      print("-"*50)
      print("SDP: {}".format(self.SDP))
      print("SDV: {}".format(self.SDV))
      print("EDV: {}".format(self.EDV))
      print("EDP: {}".format(self.EDP))
    if systole:
      print("="*50)
      print("Systole values:")
      print("-"*50)
      print("SSV: {}".format(self.SSV))
      print("SSP: {}".format(self.SSP))
      print("ESV: {}".format(self.ESV))
      print("ESP: {}".format(self.ESP))
    if computed:
      print("="*50)
      print("Computed values:")
      print("-"*50)
      # ---
      # stroke volume
      print("SV: {}".format(self.SV))
      # stroke work
      print("SW: {}".format(self.SW))
      # cardiac output
      print("CO: {}".format(self.CO))
      # coupling ratio
      print("CR: {}".format(self.CR))
      # arterial elastance
      print("Ea: {}".format(self.Ea))
      # end-systolic elastance
      print("Ees: {}".format(self.Ees))
      # Ejaction fraction
      print("EF: {}".format(self.EF))
    print("="*50)
  
  def to_dict(self):
    """Returns a dictionary object with all CD data"""
    _d = dict()
    _d["Vmin"] = self.Vmin
    _d["Vmax"] = self.Vmax
    _d["Pmin"] = self.Pmin
    _d["Pmax"] = self.Pmax
    _d["SDP"] = self.SDP
    _d["SDV"] = self.SDV
    _d["EDV"] = self.EDV
    _d["EDP"] = self.EDP
    _d["SSV"] = self.SSV
    _d["SSP"] = self.SSP
    _d["ESV"] = self.ESV
    _d["ESP"] = self.ESP
    _d["SV"] = self.SV
    _d["SW"] = self.SW
    _d["CO"] = self.CO
    _d["CR"] = self.CR
    _d["Ea"] = self.Ea
    _d["Ees"] = self.Ees
    _d["EF"] = self.EF
    return _d
  
  def to_df(self):
    """Returns a pandas dataframe object with all CD data"""
    _d = self.to_dict()

    return pd.DataFrame.from_records([_d]).astype(np.float32)
  
  def get_cds(self, cd_keys, dtype=np.float32):
    _d = self.to_dict()
    return np.array([_d[k] for k in cd_keys], dtype=dtype)