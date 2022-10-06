import biorbd
import numpy as np
import bioviz


np.set_printoptions(formatter={'float': '{: 8.3f}'.format})
q = np.array([0, np.pi/6, np.pi/6])
v = np.array([0, 0, 0])
model = biorbd.Model("3D_3dof_arm.bioMod")
model.angularMomentum(q, v).to_array()
for i in range(model.nbSegment() - 1, -1, -1):
    j = i - 1
    if j != -1:
        Xij=model.globalJCS(q, j).transpose().multiply(model.globalJCS(q, i))
        print(
        f"BIORBD X_lambda({i + 1}) = \n{biorbd.RotoTrans(Xij.rot().transpose(), Xij.trans()).to_array()}")
    else:
        print(
            f"BIORBD X_lambda({i + 1}) = \n{model.globalJCS(q, i).to_array()}")

b = bioviz.Viz("3D_3dof_arm.bioMod")
b.load_movement(q[:, np.newaxis])
b.exec()