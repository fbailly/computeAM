import bioviz
import biorbd
import matplotlib.pyplot as plt
from casadi import SX, vertcat, nlpsol
from utils import *

CHECK_MODEL = False


def Xtrans(v):
    return biorbd.RotoTrans(biorbd.Rotation(), biorbd.Vector3d(v[0], v[1], v[2]))


def applyTranspose_f(RT, f):
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


def applyAdjoint_f(RT, f):
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


def sksym(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def applyTranspose(RT, I):
    return RT @ I @ RT.transpose()

def apply(RT, v):
    mat = np.zeros((6,6))
    mat[:3, :3] = RT.rot().to_array()
    mat[3:, :3] = -RT.rot().to_array()@sksym(RT.trans().to_array())
    mat[3:, 3:] = RT.rot().to_array()
    return mat@v


def apply_f(RT, f):
    mat = np.zeros((6,6))
    mat[:3, :3] = RT.rot().to_array()
    mat[:3, 3:] = -RT.rot().to_array()@sksym(RT.trans().to_array())
    mat[3:, 3:] = RT.rot().to_array()
    return mat@f


def calcBodyComToBaseCoordinate(model, q, i):
    btbc = model.allGlobalJCS(q)[i]
    bctb_t = btbc.trans().to_array() + model.CoMbySegment(q, i).to_array()
    return biorbd.RotoTrans(btbc.rot(), biorbd.Vector3d(bctb_t[0], bctb_t[1], bctb_t[2]))


def rot(RT):
    mat_rot = biorbd.Rotation(RT[0, 0], RT[0, 1], RT[0, 2], RT[1, 0], RT[1, 1], RT[1, 2], RT[2, 0], RT[2, 1], RT[2, 2])
    vec_trans = biorbd.Vector3d(RT[0, 3], RT[1, 3], RT[2, 3])
    return biorbd.RotoTrans(mat_rot, vec_trans)



def computeAM(model, q, v):
    com = model.CoM(q).to_array()
    coms = [j.to_array() for j in model.CoMbySegment(q)]
    h = np.zeros(3)
    wi = np.zeros(3)
    for i in range(model.nbSegment()):
        cci = coms[i] - com
        Iic = model.segment(i).characteristics().inertia().to_array() - sksym(cci)@sksym(cci)
        # Ri = model.allGlobalJCS(q)[i].rot().to_array()
        wi += model.segmentAngularVelocity(q, v, i, True).to_array()
        h += Iic@wi
    return h


def newComputeAM(model, q, v):
    com = model.CoM(q).to_array()
    coms = [j.to_array() for j in model.CoMbySegment(q)]
    h = 0
    for i in range(model.nbSegment()):
        cci = coms[i] - com
        wi = model.segmentAngularVelocity(q, v, i, True).to_array()
        hi = model.segment(i).characteristics().inertia().to_array()@wi
        h += hi - sksym(cci)@sksym(cci)@wi
    return h


def adjComputeAM(model, q, v):
    com = model.CoM(q).to_array()
    coms = [j.to_array() for j in model.CoMbySegment(q)]
    chi = 0
    for i in range(model.nbSegment()):
        cci = - coms[i] + com
        owi = model.segmentAngularVelocity(q, v, i, True).to_array()
        ovlci = model.CoMdotBySegment(q, v, i).to_array()
        ovi = np.concatenate((owi, ovlci))
        hroti = (model.segment(i).characteristics().inertia().to_array())@ovi[:3]
        hlini = model.segment(i).characteristics().mass()*ovi[3:]
        cihi = np.concatenate((hroti, hlini))
        cmci = biorbd.RotoTrans(biorbd.Rotation(), biorbd.Vector3d(cci[0], cci[1], cci[2]))
        chi += apply_f(cmci, cihi)
    return chi[:3]

def rbdlComputeAM(model, q, v):
    com = model.CoM(q).to_array()
    hc = np.zeros((6, model.nbSegment()))
    htot = np.zeros((1, 6))
    for i in range(model.nbSegment()):
        hc[:3, i] = model.segment(i).characteristics().inertia().to_array()@model.segmentAngularVelocity(q, v, i, True).to_array()
        hc[3:, i] = model.segment(i).characteristics().mass()*model.CoMdotBySegment(q, v, i).to_array()
    for i in range(model.nbSegment()-1, -1, -1):
        j = i-1
        if j != -1:
            hc[:, j] = hc[:, j] + applyTranspose_f(model.globalJCS(q, j).transpose().multiply(model.globalJCS(q, i)), hc[:, i])
        else:
            htot = htot + applyTranspose_f(model.globalJCS(q, i), hc[:, i])
    htot = applyAdjoint_f(Xtrans(com), htot.squeeze())
    return htot[:3]


def compute_dh_desc(Q, V, A, L, I):
    DH = []
    for i in range(A.shape[2]-1):
        dh = compute_dAM(Q[:, i], V[:, :, i], A[:, :, i], L, I)
        DH += [dh]
    return DH


np.set_printoptions(formatter={'float': '{: 8.3f}'.format})
Tf = 1
num = int(Tf*201)
dt = Tf/num
t = np.linspace(0, Tf, num=num)
Q = np.vstack((
    # 0*np.pi/2*np.sin(2*np.pi*t),
    # 0*np.pi/2*np.cos(3*np.pi*t),
    np.pi*np.sin(2*np.pi*t),
    np.pi/2*np.sin(3*np.pi*t),
    0*np.pi/2*np.cos(4*np.pi*t),
    ))
model = biorbd.Model("3D_3dof_arm.bioMod")
v = np.diff(Q)/dt
a = np.diff(v)/dt
AM = np.zeros((3, num-2))
myAM = np.zeros((3, num-2))
myAM2 = np.zeros((3, num-2))
myAM3 = np.zeros((3, num-2))
myAM4 = np.zeros((3, num-2))
V = np.zeros((3, model.nbQ(), num-2))
A = np.zeros((3, model.nbQ(), num-2))
for i in range(num-2):
    qi = Q[:, i]
    vi = v[:, i]
    V[0, :, i] = vi
    A[0, :, i] = a[:, i]
    model.UpdateKinematicsCustom(qi)
    AM[:, i] = model.angularMomentum(qi, vi).to_array()
    # myAM[:, i] = newComputeAM(model, qi, vi).squeeze()
    # myAM2[:, i] = computeAM(model, qi, vi).squeeze()
    myAM3[:, i] = adjComputeAM(model, qi, vi).squeeze()
    myAM4[:, i] = rbdlComputeAM(model, qi, vi).squeeze()
dAM = np.diff(AM)/dt

plt.plot(AM.T, label="biorbd angular momentum")
# plt.plot(myAM.T, 'x', label="my angular momentum")
# plt.plot(myAM2.T, 'o', label="my 2nd angular momentum")
plt.plot(myAM3.T, '+', label="my 3rd angular momentum")
plt.plot(myAM4.T, '^', label="rbdl angular momentum")
# plt.plot(dAM.T, 'o', label="dAM")
plt.legend()
plt.show()

Qdecs = Q.copy()
Qdecs[1, :] = Qdecs[0, :] + Qdecs[1, :]  # sesc angle representation
Vdesc = V.copy()
Vdesc[:, 1, :] = Vdesc[:, 0, :] + Vdesc[:, 1, :]  # velocity propagation
Adesc = A.copy()
Adesc[:, 1, :] = Adesc[:, 0, :] + Adesc[:, 1, :]  # acceleration propagation

_m1 = 1.
_m2 = 1.
_l1 = 1.
_c1 = 0.5
_c2 = 0.5
_M = _m1 + _m2


def compute_l_I(CHECK_MODEL=True):
    if CHECK_MODEL:
        l1 = np.array([(_c1*_m1+_l1*_m2-_c1*_M)/_M, _c2*_m2/_M])
        L1 = np.zeros((3, 6))
        l2 = np.array([(_c1*_m1+_l1*_m2-_l1*_M)/_M, (_c2*_m2-_c2*_M)/_M])
        L2 = np.zeros((3, 6))
        I1 = np.zeros((3, 3))
        I2 = np.zeros((3, 3))
        i1 = np.array([1, 0, 0, 1, 0, 1])
        i2 = np.array([1, 0, 0, 1, 0, 1])
    else:
        l1 = SX.sym('l1', 2)
        l2 = SX.sym('l2', 2)
        i1 = SX.sym("I1", 6)
        i2 = SX.sym("I2", 6)
        I1, I2 = SX(3, 3), SX(3, 3)
        L1, L2 = SX(3, 6), SX(3, 6)
    return L1, L2, l1, l2, I1, I2, i1, i2


(L1, L2, l1, l2, I1, I2, i1, i2) = compute_l_I(CHECK_MODEL)
build_L(L1, l1)
build_L(L2, l2)
fill_inertia(I1, i1)
fill_inertia(I2, i2)
L = [L1, L2]
I = [I1, I2]
DH = compute_dh_desc(Qdecs, Vdesc, Adesc, L, I)
obj = 0
for i in range(len(DH)):
    err = dAM.T[i] - DH[i]
    obj += err.T @ err

if CHECK_MODEL:
    print(f"Error on dHg={np.sqrt(obj)}")
    plt.plot(dAM.T[:] - DH[:], label="error")
    plt.legend()
    plt.figure()
    plt.plot(dAM.T[:])
    plt.plot(DH[:], 'x')
    plt.show()
else:
    # Form the NLP
    X = vertcat(l1, l2, i1, i2)
    g = X
    nlp = {'x': X, 'f': obj, 'g': g}

    # Pick an NLP solver
    MySolver = "ipopt"

    # Solver options
    opts = {}
    opts["ipopt.linear_solver"] = "ma57"
    opts["ipopt.max_iter"] = 1000

    # Allocate a solver
    solver = nlpsol("solver", MySolver, nlp, opts)

    # Solve the NLP
    lbg = np.array([0.1, 0.1,  # L1
                    -0.4, -0.4,  # L2
                    0.4, -0.4, -0.4, 0.4, -0.4, 0.4,  # I1
                    0.4, -0.4, -0.4, 0.4, -0.4, 0.4,  # I2
                    ])
    ubg = np.array([0.4, 0.4,  # L1
                    -0.1, -0.1,  # L2
                    1.6, 0.4, 0.4, 1.6, 0.4, 1.6,  # I1
                    1.6, 0.4, 0.4, 1.6, 0.4, 1.6,  # I2
                    ])
    sol = solver(lbg=lbg, ubg=ubg)

    # Retrieve solution
    l1_sol = sol['x'][:2]
    l2_sol = sol['x'][2:4]
    i1_sol = sol['x'][4:10]
    i2_sol = sol['x'][10:16]
    L1_sol = np.zeros((3, 6))
    L2_sol = np.zeros((3, 6))
    I1_sol = np.zeros((3, 3))
    I2_sol = np.zeros((3, 3))
    build_L(L1_sol, l1_sol)
    build_L(L2_sol, l2_sol)
    fill_inertia(I1_sol, i1_sol)
    fill_inertia(I2_sol, i2_sol)
    L_sol = [L1_sol, L2_sol]
    I_sol = [I1_sol, I2_sol]
    DH = compute_dh_desc(Qdecs, Vdesc, Adesc, L_sol, I_sol)
    obj = 0
    for i in range(len(DH)):
        err = dAM.T[i] - DH[i]
        obj += err.T@err
    print(f"Error on dHg={np.sqrt(obj)}")
    # Plot solution
    (L1, L2, l1, l2, I1, I2, i1, i2) = compute_l_I()
    print(f"l1_sol={l1_sol}, l1_orig={l1}")
    print(f"l2_sol={l2_sol}, l2_orig={l2}")
    print(f"i1_sol={i1_sol}, i1_orig={i1}")
    print(f"i2_sol={i2_sol}, i2_orig={i2}")
    plt.plot(dAM.T[:] - DH[:], label="error after optim")
    plt.legend()
    plt.figure()
    plt.plot(dAM.T[:], label="Model DAM")
    plt.plot(DH[:], 'x', label="DESC DAM")
    plt.legend()
    plt.show()




b = bioviz.Viz("3D_3dof_arm.bioMod")
b.load_movement(Q)
b.exec()