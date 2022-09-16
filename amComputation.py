import biorbd
import numpy as np
import matplotlib.pyplot as plt
import bioviz


def sksym(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def Xtrans(v):
    return biorbd.RotoTrans(biorbd.Rotation(), biorbd.Vector3d(v[0], v[1], v[2]))


def applyTranspose_f(RT, f):
    '''copied from https://rbdl.github.io/d4/d41/_spatial_algebra_operators_8h_source.html#l00179'''
    E = RT.rot().to_array()
    r = RT.trans().to_array()
    E_T_f = np.array([
    E[0, 0] * f[3] + E[1, 0] * f[4] + E[2, 0] * f[5],
    E[0, 1] * f[3] + E[1, 1] * f[4] + E[2, 1] * f[5],
    E[0, 2] * f[3] + E[1, 2] * f[4] + E[2, 2] * f[5]
    ])
    return np.array([
        E[0, 0] * f[0] + E[1, 0] * f[1] + E[2, 0] * f[2] - r[2] * E_T_f[1] + r[1] * E_T_f[2],
        E[0, 1] * f[0] + E[1, 1] * f[1] + E[2, 1] * f[2] + r[2] * E_T_f[0] - r[0] * E_T_f[2],
        E[0, 2] * f[0] + E[1, 2] * f[1] + E[2, 2] * f[2] - r[1] * E_T_f[0] + r[0] * E_T_f[1],
        E_T_f[0],
        E_T_f[1],
        E_T_f[2]
        ])


def applyTranspose_rbi(RT, I, com, m):
    E = RT.rot().to_array()
    r = RT.trans().to_array()
    E_T_mr = E.transpose()@com + m*r
    Inew = E.transpose()@I@E - sksym(r)@sksym(E.transpose()@com) - sksym(E_T_mr)@sksym(r)
    return Inew, E_T_mr, m




def applyAdjoint_f(RT, f):
    '''copied from https://rbdl.github.io/d4/d41/_spatial_algebra_operators_8h_source.html#l00233'''
    E = RT.rot().to_array()
    r = RT.trans().to_array()
    En_rxf = E@(np.array([f[0], f[1], f[2]]) - np.cross(r, np.array([f[3], f[4], f[5]])))
    return np.array([
        En_rxf[0],
        En_rxf[1],
        En_rxf[2],
        E[0, 0] * f[3] + E[0, 1] * f[4] + E[0, 2] * f[5],
        E[1, 0] * f[3] + E[1, 1] * f[4] + E[1, 2] * f[5],
        E[2, 0] * f[3] + E[2, 1] * f[4] + E[2, 2] * f[5]
        ])


def rbdlComputeAM(model, q, v):
    '''copied from https://rbdl.github.io/d4/d22/rbdl__utils_8cc_source.html#l00190'''
    Ic = np.zeros((3, 3, model.nbSegment()))
    c = np.zeros((3, model.nbSegment()))
    m = np.zeros((1, model.nbSegment()))
    hc = np.zeros((6, model.nbSegment()))
    htot = np.zeros((1, 6))
    Itot = np.zeros((3, 3))
    ctot = np.zeros((1, 3))
    mtot = np.zeros((1))
    for i in range(model.nbSegment()):
        Ic[:, :, i] = model.segment(i).characteristics().inertia().to_array()
        m[:, i] = model.segment(i).characteristics().mass()
        c[:, i] = model.segment(i).characteristics().CoM().to_array()
        hc[:3, i] = Ic[:, :, i]@model.segmentAngularVelocity(q, v, i, True).to_array()
        hc[3:, i] = m[:, i]*model.CoMdotBySegment(q, v, i).to_array()
    for i in range(model.nbSegment()-1, -1, -1):
        j = i-1
        if j != -1:
            hc[:, j] = hc[:, j] + applyTranspose_f(model.globalJCS(q, j).transpose().multiply(model.globalJCS(q, i)), hc[:, i])
            Ii, comi, mi = applyTranspose_rbi(model.globalJCS(q, j).transpose().multiply(model.globalJCS(q, i)), Ic[:, :, i], c[:, i], m[:, i])
            Ic[:, :, j] += Ii
            c[:, j] += comi.squeeze()
            m[:, j] += mi
        else:
            htot = htot + applyTranspose_f(model.globalJCS(q, i), hc[:, i])
            Ii, comi, mi = applyTranspose_rbi(model.globalJCS(q, i), Ic[:, :, i], c[:, i], m[:, i])
            Itot += Ii
            ctot += comi
            mtot += mi

    com = (ctot/mtot).squeeze()
    htot1 = applyAdjoint_f(Xtrans(com), htot.squeeze())
    return htot1[:3]


Tf = 1
num = int(Tf*100)
dt = Tf/num
t = np.linspace(0, Tf, num=num)
Q = np.vstack((
    np.pi*np.sin(2*np.pi*t),
    0*np.pi/2*np.sin(3*np.pi*t),
    0*np.pi/2*np.cos(4*np.pi*t),
    ))
model = biorbd.Model("3D_3dof_arm.bioMod")
v = np.diff(Q)/dt
biorbdAM = np.zeros((3, num-2))
myAM = np.zeros((3, num-2))
V = np.zeros((3, model.nbQ(), num-2))
for i in range(num-2):
    qi = Q[:, i]
    vi = v[:, i]
    V[0, :, i] = vi
    model.UpdateKinematicsCustom(qi)
    biorbdAM[:, i] = model.angularMomentum(qi, vi).to_array()
    myAM[:, i] = rbdlComputeAM(model, qi, vi)

plt.plot(biorbdAM.T, label="biorbd angular momentum")
plt.plot(myAM.T, 'x', label="rbdl angular momentum")
plt.legend()
plt.show()

b = bioviz.Viz("3D_3dof_arm.bioMod")
b.load_movement(Q)
b.exec()