import bioviz
import biorbd
import matplotlib.pyplot as plt
from casadi import SX, vertcat, nlpsol
from utils import *

CHECK_MODEL = False


def sksym(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def computeAM(model, q, v):
    com = model.CoM(q).to_array()
    coms = [j.to_array() for j in model.CoMbySegment(q)]
    h = 0
    for i in range(model.nbSegment()):
        cci = coms[i] - com
        Iic = model.segment(i).characteristics().inertia().to_array() - sksym(cci)@sksym(cci)
        # Ri = model.allGlobalJCS(q)[i].rot().to_array()
        if i > 0:
            wi = model.segmentAngularVelocity(q, v, i, True).to_array() + model.segmentAngularVelocity(q, v, i-1, True).to_array()
        else:
            wi = model.segmentAngularVelocity(q, v, i, True).to_array()
        h += Iic@wi
    return h


def newcomputeAM(model, q, v):
    com = model.CoM(q).to_array()
    coms = [j.to_array() for j in model.CoMbySegment(q)]
    h = 0
    for i in range(model.nbSegment()):
        cci = coms[i] - com
        wi = model.segmentAngularVelocity(q, v, i, True).to_array()
        hi = model.segment(i).characteristics().inertia().to_array()@wi
        h += hi - sksym(cci)@sksym(cci)@wi
    return h


def compute_dh_desc(Q, V, A, L, I):
    DH = []
    for i in range(A.shape[2]-1):
        dh = compute_dAM(Q[:, i], V[:, :, i], A[:, :, i], L, I)
        DH += [dh]
    return DH

Tf = 1
num = int(Tf*200)
dt = Tf/num
t = np.linspace(0, Tf, num=num)
Q = np.vstack((0*np.pi/2*np.sin(2*np.pi*t),
               0*np.pi/2*np.cos(3*np.pi*t),
               0*np.pi*np.cos(2*np.pi*t),
               np.pi/2*np.sin(3*np.pi*t),
               0*np.pi/2*np.cos(4*np.pi*t),
               ))
model = biorbd.Model("3D_3dof_arm.bioMod")
v = np.diff(Q)/dt
a = np.diff(v)/dt
AM = np.zeros((3, num-2))
myAM = np.zeros((3, num-2))
myAM2 = np.zeros((3, num-2))
V = np.zeros((3, 5, num-2))
A = np.zeros((3, 5, num-2))
for i in range(num-2):
    qi = Q[:, i]
    vi = v[:, i]
    V[0, :, i] = vi
    A[0, :, i] = a[:, i]
    model.UpdateKinematicsCustom(qi)
    AM[:, i] = model.angularMomentum(qi, vi).to_array()
    myAM[:, i] = newcomputeAM(model, qi, vi).squeeze()
    myAM2[:, i] = computeAM(model, qi, vi).squeeze()
dAM = np.diff(AM)/dt

plt.plot(AM.T, label="angular momentum biorbd")
plt.plot(myAM2.T, 'o', label="my 2nd angular momentum")
plt.plot(myAM.T, 'x', label="my angular momentum")
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