
import numpy as np
# Matplotlib / plot
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.spatial import Delaunay as delaunay
import sympy as sp

################## Modèle de fonction d'interpolation ##################

def interp_xxx(x_obs, y_obs, z_obs, x_int, y_int):

    # Interpolation par ???
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    pass
    # ...
    #
    # return z_int

####################### Fonctions d'interpolation ######################

def interp_lin(x_obs, y_obs, z_obs, x_int, y_int):

    # Interpolation par ???
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    z_int = np.nan*np.zeros(x_int.shape)
    tri = delaunay(np.hstack((x_obs, y_obs)))
    for i in np.arange(0,x_int.shape[0]):
        for j in np.arange(0,x_int.shape[1]):

            # on recherche le numéro du triangle dans tri contenant le point (x0,y0)
            idx_t = tri.find_simplex(np.array([x_int[i,j], y_int[i,j]]))
            if idx_t == -1 : continue
            # on récupère les numéros des sommets du triangle contenant le point (x0,y0)
            idx_s = tri.simplices[idx_t, :]
            # x_obs, y_obs sont des tableaux à 2 dimensions ; il faut les préciser pour en extraire un scalaire
            x1, y1, z1 = x_obs[idx_s[0]].item(), y_obs[idx_s[0]].item(), z_obs[idx_s[0]].item()
            x2, y2, z2 = x_obs[idx_s[1]].item(), y_obs[idx_s[1]].item(), z_obs[idx_s[1]].item()
            x3, y3, z3 = x_obs[idx_s[2]].item(), y_obs[idx_s[2]].item(), z_obs[idx_s[2]].item()
            mat_A = np.array([[x1, y1, 1],
                              [x2, y2, 1],
                              [x3, y3, 1]])
            mat_B = np.array([z1, z2, z3])
            mat_X = np.linalg.solve(mat_A, mat_B)
            z_int[i, j] = mat_X[0] * x_int[i, j] + mat_X[1] * y_int[i, j] + mat_X[2]

    return z_int


def interp_spline(x_obs, y_obs, z_obs, x_int, y_int, rho=0):
    n = len(x_obs)

    # Construction de la matrice A
    A = np.zeros((n + 3, n + 3))

    # Remplissage de la matrice A
    A[:n, 0] = 1
    A[:n, 1] = x_obs.flatten()
    A[:n, 2] = y_obs.flatten()

    for i in range(n):
        for j in range(n):
            r = np.sqrt((x_obs[i] - x_obs[j]) ** 2 + (y_obs[i] - y_obs[j]) ** 2)
            if i == j :
                A[i, j + 3] = rho
            else :
                r * np.log(r + 1e-10)
    A[n, 3:] = 1
    A[n + 1, 3:] = x_obs.flatten()
    A[n + 2, 3:] = y_obs.flatten()

    # Construction et remplissage de la matrice B
    B = np.zeros(n + 3)
    B[:n] = z_obs.flatten()


    # Résolution du système
    mat_N = A.T @ A
    mat_K = A.T @ B
    mat_C = np.linalg.inv(mat_N) @ mat_K
    # mat_C = np.linalg.solve(A, B)

    # Interpolation sur la grille
    z_int = np.zeros_like(x_int)
    for i in range(x_int.shape[0]):
        for j in range(x_int.shape[1]):
            x = x_int[i, j]
            y = y_int[i, j]
            r = np.sqrt((x - x_obs) ** 2 + (y - y_obs) ** 2)
            z_int[i, j] = (mat_C[0] + mat_C[1] * x + mat_C[2] * y + np.sum(mat_C[3:] * r * np.log(r + 1e-10)))

    return z_int

def interp_ppv(x_obs, y_obs, z_obs, x_int, y_int):
    # Interpolation par plus proche voisin
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]

    z_int = np.nan*np.zeros(x_int.shape)
    for i in np.arange(0,x_int.shape[0]):
        for j in np.arange(0,x_int.shape[1]):
            d = np.sqrt((x_int[i,j]-x_obs)**2+(y_int[i,j]-y_obs)**2)
            idx = np.argmin(d)
            z_int[i,j] = z_obs[idx]
    return z_int


# def inter_inv_dist(x_obs, y_obs, z_obs, x_int, y_int, p=2, nb_pts=-1):
#     z_int = np.nan * np.zeros(x_int.shape)
#     # d = np.zeros((x_obs.shape,1))
#     for i in np.arange(0, x_int.shape[0]):
#         for j in np.arange(0, x_int.shape[1]):
#             d = np.sqrt((x_int[i, j] - x_obs) ** 2 + (y_int[i, j] - y_obs) ** 2)
#             idx = np.argsort(d, axis=0)
#             if nb_pts > 0: idx = idx[:nb_pts]
#             if d[i] == 0:
#                 z_int[i, j] = z_obs[i]
#             else:
#                 z_int[i, j] = (np.sum(z_obs[idx, 0] / d[idx, 0] ** p) / np.sum(1 / d[idx, 0] ** p))
#     return z_int


def interp_inv(x_obs, y_obs, z_obs, x_grd, y_grd, p=2, dmax=None):
    z_grd = np.full(x_grd.shape, np.nan)  # Initialisation avec NaN

    for i in range(x_grd.shape[0]):
        for j in range(x_grd.shape[1]):
            # Coordonnées du point de la grille
            x, y = x_grd[i, j], y_grd[i, j]

            # Calcul des distances aux points d'observation
            distances = np.sqrt((x_obs - x) ** 2 + (y_obs - y) ** 2)

            # Filtrer les points selon la distance maximale si définie
            if dmax is not None:
                mask = distances <= dmax
                distances = distances[mask]
                values = z_obs[mask]
            else:
                values = z_obs

            # Éviter les divisions par zéro
            if len(distances) == 0 or np.any(distances == 0):
                z_grd[i, j] = z_obs[np.argmin(distances)] if np.any(distances == 0) else np.nan
                continue

            # Calcul de la pondération
            weights = 1 / distances ** p
            z_grd[i, j] = np.sum(weights * values) / np.sum(weights)

    return z_grd


############################# Visualisation ############################

def plot_contour_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé du champ interpolé sous forme d'isolignes
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)

    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    plt.contour(x_grd, y_grd, z_grd_m, int(np.round((np.max(z_grd_m)-np.min(z_grd_m))/4)),colors ='k')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        dx = max(x_obs)-min(x_obs)
        dy = max(y_obs)-min(y_obs)
        minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
        miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    else:
        dx = np.max(x_grd)-np.min(x_grd)
        dy = np.max(y_grd)-np.min(y_grd)
        minx = np.min(x_grd)-0.05*dx; maxx = np.max(x_grd)+0.05*dx
        miny = np.min(y_grd)-0.05*dy; maxy = np.max(y_grd)+0.05*dy
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_surface_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), minmax = [0,0], xlabel = "", ylabel = "", zlabel = "", title = "", fileo = "", cmap = cm.terrain):
    # Tracé du champ interpolé sous forme d'une surface colorée
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # minmax : valeurs min et max de la variable interpolée (facultatif)
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    # cmap : nom de la carte de couleur
    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    if minmax[0] < minmax[-1]:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cmap, vmin = minmax[0], vmax = minmax[-1], shading = 'auto')
    else:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cmap, shading = 'auto')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        dx = max(x_obs)-min(x_obs)
        dy = max(y_obs)-min(y_obs)
        minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
        miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    else:
        dx = np.max(x_grd)-np.min(x_grd)
        dy = np.max(y_grd)-np.min(y_grd)
        minx = np.min(x_grd)-0.05*dx; maxx = np.max(x_grd)+0.05*dx
        miny = np.min(y_grd)-0.05*dy; maxy = np.max(y_grd)+0.05*dy
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_points(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)

    fig = plt.figure()
    ax = plt.gca()
    plt.plot(x_obs, y_obs, 'ok', ms = 4)
    dx = max(x_obs)-min(x_obs)
    dy = max(y_obs)-min(y_obs)
    minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
    miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_patch(x_obs, y_obs, z_obs, fig = "", minmax = [0,0], xlabel = "", ylabel = "", zlabel = "", title = "", fileo = "", cmap = cm.terrain, marker = 'o', s= 80,ec=None,lw=0, cb=True):
    # Tracé des valeurs observées
    # x_obs, y_obs, z_obs : observations
    # fig : figure sur laquelle faire le tracé (facultatif)
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    # cmap : nom de la carte de couleur
    # marker : type de marker
    # s : taille du marker
    # ec : couleur du contour des marker
    # lw : taille du contour des marker

    if fig == "": fig = plt.figure()
    if minmax[0] < minmax[-1]:
      p=plt.scatter(x_obs, y_obs, marker = marker, c = z_obs, s = s, cmap=cmap, vmin = minmax[0], \
      vmax = minmax[-1], edgecolor = ec, linewidth=lw)
    else:
      p=plt.scatter(x_obs, y_obs, marker = marker, c = z_obs, s = s, cmap=cmap, edgecolor = ec, linewidth=lw)
    dx = max(x_obs)-min(x_obs)
    dy = max(y_obs)-min(y_obs)
    minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
    miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if cb:
      fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_triangulation(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé de la triangulation sur des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    from scipy.spatial import Delaunay as delaunay
    tri = delaunay(np.hstack((x_obs,y_obs)))

    plt.figure()
    plt.triplot(x_obs[:,0], y_obs[:,0], tri.simplices)
    plt.plot(x_obs, y_obs, 'or', ms=4)
    dx = max(x_obs)-min(x_obs)
    dy = max(y_obs)-min(y_obs)
    minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
    miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def calc_nuee(x_obs, y_obs, z_obs,):
    g_raw = []
    h_raw  = []
    for i in np.arange(x_obs.shape[0]):
        for j in np.arange(x_obs.shape[0]):
            d = np.sqrt((x_obs[i,0]- x_obs[j,0])**2 + (y_obs[i,0]- y_obs[j,0])**2)
            nue = 0.5 *(z_obs[i,0] - z_obs[j,0])**2
            g_raw.append(nue)
            h_raw.append(d)

    return h_raw, g_raw


def calc_var_exp(x_obs,y_obs,z_obs,hmax=160,nmax=500, pas=30):
    g_raw = []
    h_raw = []
    for i in np.arange(x_obs.shape[0]):
        for j in np.arange(x_obs.shape[0]):
            d = np.sqrt((x_obs[i, 0] - x_obs[j, 0]) ** 2 + (y_obs[i, 0] - y_obs[j, 0]) ** 2)
            nue = 0.5 * (z_obs[i, 0] - z_obs[j, 0]) ** 2
            g_raw.append(nue)
            h_raw.append(d)

    g_raw = np.array(g_raw)
    h_raw = np.array(h_raw)


    h = np.arange(0, hmax, pas)
    g = np.nan*np.zeros(h.shape)
    for k in np.arange(len(h)):
        mask =  (h_raw >= h[k]) & (h_raw < h[k]+pas)
        g[k] = np.mean(g_raw[mask])


    return h, g

def fit_var_ana(x_obs,y_obs,z_obs,hmax=160,nmax = 500,model="cub"):
    h_raw, g_raw = calc_var_exp(x_obs,y_obs,z_obs)
    C, a, h = sp.symbols('C, a, h')
    y = C * (7 * (h**2 / a**2) -(35/4)*(h**3 / a**3) + (7/2)*(h**5 / a**5) -(3/4)*(h/a)**7)
    if model == "cub":

        mat_A = np.zeros((len(h_raw),2))
        mat_B = np.zeros((len(h_raw),1))
        diff_C = sp.diff(y, C)
        diff_a = sp.diff(y, a)
        dc, da = float('inf'), float('inf')
        C0, a0 = float(g_raw.max()), float(h_raw[np.argmax(g_raw)])
        while abs(dc) > 0.1 or abs(da) > 0.1:
            for i in range(len(h_raw)):
                if h_raw[i] <= a0 :

                    g0 = sp.lambdify([C, a, h], y)
                    A0 = sp.lambdify([C, a, h], diff_C)
                    A1 = sp.lambdify([C, a, h], diff_a)

                    g = g_raw[i] - g0(C0, a0, h_raw[i])
                    mat_B[i] = g
                    mat_A[i,0] = A0(C0, a0, h_raw[i])
                    mat_A[i,1] = A1(C0, a0, h_raw[i])

                else:
                    mat_B[i] = C0
                    mat_A[i, 0] = 1
                    mat_A[i, 1] = 0

            mat_N = mat_A.T@mat_A
            mat_K = mat_A.T@mat_B
            mat_X = np.linalg.inv(mat_N)@mat_K
            dc, da = mat_X[0, 0], mat_X[1,0]
            C0, a0 = dc + C0, da + a0
        return  float(C0), float(a0)
    else:
        mat_A = h_raw
        mat_B = g_raw
        mat_N = mat_A.T@mat_A
        mat_K = mat_A.T@mat_B
        mat_W = (1/mat_N)*mat_K
        return float(mat_W)


def calc_va_ana(h_exp, c, a=0, model="cub"):
    if model == "cub":
        gamma_cub = c * (7 * (h_exp ** 2 / a ** 2) - (35 / 4) * (h_exp ** 3 / a ** 3) + (7 / 2) * (h_exp ** 5 / a ** 5) - (3 / 4) * (h_exp / a) ** 7)
        return gamma_cub
    else:
        gamma_lin = c*h_exp
        return gamma_lin

def interp_krg(x_obs, y_obs, z_obs, x_int, y_int, c, a=0, kind_var="cub", r_maw_var=160, card_var=500):
    n = x_obs.shape[0]
    mat_A = np.zeros((n + 1, n + 1))
    mat_B = np.zeros((n + 1, 1))

    # Remplir la matrice A avec les valeurs du variogramme
    for i in range(n):
        for j in range(n):
            d = np.sqrt((x_obs[i, 0] - x_obs[j, 0]) ** 2 + (y_obs[i, 0] - y_obs[j, 0]) ** 2)
            mat_A[i, j] = calc_va_ana(d, c, a, model=kind_var)

    mat_A[:-1, -1] = 1
    mat_A[-1, :-1] = 1
    mat_A[-1, -1] = 0



    z_int = np.nan * np.zeros(x_int.shape)
    sigma = np.nan * np.zeros(x_int.shape)

    for i in range(x_int.shape[0]):
        for j in range(x_int.shape[1]):
            # Remplir la matrice B avec les valeurs du variogramme pour le point (x_int[i,j], y_int[i,j])
            d = np.sqrt((x_int[i, j] - x_obs[:, 0]) ** 2 + (y_int[i, j] - y_obs[:, 0]) ** 2)
            mat_B[:-1, 0] = calc_va_ana(d, c, a, model=kind_var)
            mat_B[-1, 0] = 1  # Contrainte de Lagrange

            # Résolution du système linéaire pour obtenir les poids lambda
            lambda_vec = np.linalg.solve(mat_A, mat_B)

            # Calcul de la valeur interpolée
            z_int[i, j] = np.sum(lambda_vec[:-1] * z_obs)

            # Calcul de l'incertitude σ²(s)
            sigma[i, j] = np.sum(lambda_vec[:-1] * mat_B[:-1, 0]) + lambda_vec[-1]

    return z_int, sigma




























