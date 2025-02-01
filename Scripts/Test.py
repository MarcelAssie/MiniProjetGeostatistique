import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import lib_gs as lib

obs_mf =np.loadtxt('../Donnees/obs_wind_mf.txt')
E_mf = obs_mf[:,0:1]/1e3 # km
N_mf = obs_mf[:,1:2]/1e3 # km
v_mf = obs_mf[:,2:3]

sortie = lib.cross_validation(E_mf, N_mf, v_mf, "lineaire")
print(sortie)