import numpy as np


def fill_inertia(I, i):
    I[0, 0] = i[0]
    I[0, 1] = i[1]
    I[0, 2] = i[2]
    I[1, 0] = i[1]
    I[1, 1] = i[3]
    I[1, 2] = i[4]
    I[2, 0] = i[2]
    I[2, 1] = i[4]
    I[2, 2] = i[5]
    return


def build_L(L, l):
    L[0, 0] = l[0]
    L[0, 1] = l[1]
    L[1, 2] = l[0]
    L[1, 3] = l[1]
    L[2, 4] = l[0]
    L[2, 5] = l[1]
    return


def compute_dAM(q, v, a, L, I):
    th, dth = compute_th_dth(q, v)
    thdth = - dth@th.T - th@dth.T
    dc1cx2 = L[0] @ thdth @ L[0].T
    dc2cx2 = L[1] @ thdth @ L[1].T
    dh1 = (I[0] + L[0] @ th @ th.T @ L[0].T) @ a[:, 0] - dc1cx2 @ v[:, 0]
    dh2 = (I[1] + L[1] @ th @ th.T @ L[1].T) @ a[:, 1] - dc2cx2 @ v[:, 1]
    return dh1 + dh2

def compute_th_dth(q, v):
    th = np.zeros((6, 3))
    dth = np.zeros((6, 3))
    sth1 = np.sin(q[0])
    dsth1 = v[0, 0]*np.sin(q[0])
    cth1 = np.cos(q[0])
    dcth1 = v[0, 0]*np.cos(q[0])
    sth2 = np.sin(q[1])
    dsth2 = v[0, 1]*np.sin(q[1])
    cth2 = np.cos(q[1])
    dcth2 = v[0, 1]*np.cos(q[1])
    th[0, 1] = -sth1
    th[0, 2] = cth1
    th[1, 1] = -sth2
    th[1, 2] = cth2
    th[2, 0] = sth1
    th[3, 0] = sth2
    th[4, 0] = -cth1
    th[5, 0] = -cth2
    dth[0, 1] = -dcth1
    dth[0, 2] = -dsth1
    dth[1, 1] = -dcth2
    dth[1, 2] = -dsth2
    dth[2, 0] = dcth1
    dth[3, 0] = dcth2
    dth[4, 0] = dsth1
    dth[5, 0] = dsth2
    return th, dth
