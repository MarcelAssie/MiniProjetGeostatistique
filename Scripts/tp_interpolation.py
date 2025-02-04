#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
#    TP - Introduction à l'interpolation spatiale et aux géostatistiques #
##########################################################################

# P. Bosser / ENSTA Bretagne
# Version du 13/03/2022


# Numpy
import numpy as np
# Matplotlib / plot
import matplotlib.pyplot as plt
# lib geostatistic
import lib_gs as gs

path = '../Donnees/'

obs_mf =np.loadtxt(f"{path}obs_wind_mf.txt")
x_obs = obs_mf[:,0:1]/1e3 # km
y_obs = obs_mf[:,1:2]/1e3 # km
z_obs = obs_mf[:,2:3]


## Sorties du champ "vent en rafale" de la réanalyse ERA5
obs_era5 = np.loadtxt(f"{path}mod_wind_era5.txt")
E_era5 = obs_era5[:, 0:1] / 1e3  # km
N_era5 = obs_era5[:, 1:2] / 1e3  # km
v_era5 = obs_era5[:, 2:3]

# Contour de la France métropolitaine
FR_contour =np.loadtxt(f"{path}FR_contour.txt")
E = FR_contour[:,0:1]/1e3
N = FR_contour[:,1:2]/1e3

# # Visualisation des données en entrée : sites de mesure
# #gs.plot_points(x_obs, y_obs, xlabel = 'x [m]', ylabel = 'y [m]', title = "sites d'observation")
# # Visualisation des données en entrée : sites de mesure coloré en fonction de la VR
# #gs.plot_patch(x_obs, y_obs, z_obs, xlabel = 'x [m]', ylabel = 'y [m]', zlabel = 'z [m]', title = "observations")
#
# # Création d'une grille planimétrique pour l'interpolation
x_grd, y_grd = np.meshgrid(np.linspace(np.floor(np.min(x_obs)), np.ceil(np.max(x_obs)), 100), np.linspace(np.floor(np.min(y_obs)), np.ceil(np.max(y_obs)), 100))
#
# # Interpolation linéaire
z_grd_lin = gs.interp_lin(x_obs, y_obs, z_obs, x_grd, y_grd)

# Interpolation spline
#z_grd_spline = gs.interp_spline(x_obs, y_obs, z_obs, x_grd, y_grd)

# # Interpolation par plus proche voisin (PPV)
# #z__grd_int = gs.interp_ppv(x_obs, y_obs, z_obs, x_grd, y_grd)
#
# z_grd_inv = gs.inter_inv_dist(x_obs, y_obs, z_obs, x_grd, y_grd, nb_pts=15)
#
# Visualiation de l'interpolation PPV : lignes de niveau
# gs.plot_contour_2d(x_grd, y_grd, z_grd_spline, x_obs, y_obs, xlabel = 'x [m]', ylabel = 'y [m]', title = 'PPV')
#
# # Visualiation de l'interpolation INV : surface colorée
# #gs.plot_surface_2d(x_grd, y_grd, z_grd_inv, x_obs, y_obs, xlabel = 'x [m]', ylabel = 'y [m]', title = 'INB')
#
# Interpolation en un point de l'espace

# zi = gs.inter_inv_dist(x_obs, y_obs, z_obs, np.array([[225]]),  np.array([[180]]))
# print("La valeur interpolée en (225,180) est "+str(zi))
# # plt.show()

# h_raw, g_raw = gs.calc_nuee(x_obs, y_obs, z_obs)
# plt.plot(h_raw,g_raw,'.')
# plt.xlabel(r'$h_{i,j}$')
# plt.ylabel(r'$\Delta z_{i,j}^2/2$')
# plt.grid()
#


h_exp, g_exp = gs.calc_var_exp(x_obs,y_obs,z_obs,hmax=160,nmax=500)
# plt.plot(h_raw,g_raw,'.',label='Nuéee')
# plt.plot(h_exp,g_exp,'r',label='Var. exp.')
# plt.xlabel(r'$h$')
# plt.ylabel(r'$\Delta z_i^2/2$')
# plt.legend()
# plt.grid()

c,a = gs.fit_var_ana(x_obs,y_obs,z_obs,hmax=160,nmax = 500,model="cub")
gamma_cub = gs.calc_va_ana(h_exp, c=c, a=a, model="cub")
print("Paramètres du variogramme cubique:",a,c)
c = gs.fit_var_ana(x_obs,y_obs,z_obs,hmax=160,nmax = 500,model="lin")
gamma_lin = gs.calc_va_ana(h_exp, c=c,model="lin")
print("Paramètre du variogramme linéaire:",c)
plt.plot(h_exp,g_exp,'r',label='Var. exp.')
plt.plot(h_exp,gamma_cub,'g',label=r'$\gamma_{cub}$')
plt.plot(h_exp,gamma_lin,'b',label=r'$\gamma_{lin}$')
plt.xlabel(r'$h$')
plt.ylabel(r'$\Delta z_i^2/2$')
plt.legend()
plt.grid()


plt.show()

# print(gs.fit_var_ana(x_obs,y_obs,z_obs,hmax=160,nmax = 500,model="cub"))
#
# z_int = gs.interp_krg(x_obs, y_obs, z_obs, x_grd, y_grd,c=c, a=a, kind_var="cub", r_maw_var=160, card_var=500)
# print(z_int)

# validation = gs.cross_validation(x_obs, y_obs, method="lineaire")
# print(validation)