import numpy as np

import lib_gs as lib
import sympy as sp
def fit_var_ana(x_obs,y_obs,z_obs,hmax=160,nmax = 500,model="cub"):
    h_raw, g_raw = lib.calc_var_exp(x_obs,y_obs,z_obs)
    C, a, h = sp.symbols('C a h')
    y = C * (1- sp.exp(-(h/a)))
    if model == "cub":

        mat_A = np.zeros((len(h_raw),2))
        mat_B = np.zeros((len(h_raw),1))
        diff_C = sp.diff(y, C)
        diff_a = sp.diff(y, a)
        dc, da = float('inf'), float('inf')
        C0, a0 = g_raw.max(), h_raw[np.argmax(g_raw)]
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

            print(mat_A)
            print(mat_B)
            mat_N = mat_A.T@mat_A
            mat_K = mat_A.T@mat_B
            try:
                mat_X = np.linalg.solve(mat_N, mat_K)
            except np.linalg.LinAlgError:
                return None  # Retourne None en cas de problème d'inversion
            # mat_X = np.linalg.inv(mat_N)@mat_K
            # Mise à jour des paramètres
            dc, da = mat_X[0, 0], mat_X[1, 0]
            print(dc, da)
            C0 += dc
            a0 += da
            print(f"C0 = {C0} et a0 = {a0}")
        return  float(C0), float(a0)
    else:
        mat_A = h_raw
        mat_B = g_raw
        mat_N = mat_A.T@mat_A
        mat_K = mat_A.T@mat_B
        mat_W = (1/mat_N)*mat_K
        return float(mat_W)


if __name__ == '__main__':
    ## Mesures de "vent en rafale" du réseau de stations sol
    obs_mf = np.loadtxt("../Donnees/obs_wind_mf.txt")
    E_mf = obs_mf[:, 0:1] / 1e3  # km
    N_mf = obs_mf[:, 1:2] / 1e3  # km
    v_mf = obs_mf[:, 2:3]
    print(fit_var_ana(E_mf, N_mf, v_mf, model="cub"))