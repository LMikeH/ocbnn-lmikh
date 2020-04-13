import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from bnn.svgd import *
from bnn.hmc import *


def rastrigin(params, noise_factor):
    """
    This function generates
    :param params:
    :param noise_factor:
    :return:
    """

    ras = 10*params.shape[1]

    for d in range(params.shape[1]):
        ras += params[:, d]**2 - 10*torch.cos(2*np.pi*params[:, d])

    y = torch.from_numpy(ras.numpy() + noise_factor*(2*np.random.rand(params.shape[0]) - 1)).float()
    return (y-y.mean())/y.std()


n_data = 50
noise = 0.00001

x = torch.from_numpy(2*5.12*np.random.rand(n_data, 2) - 5.12).float()
y = rastrigin(x, noise).reshape(-1, 1)


dict = {'dataset_name': "2d-test", "X_train": x/torch.max(x), "Y_train": y}


logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s: %(message)s")
logging.info("Running OC-BNN library...")
bnn = BNNSVGDRegressor(uid='2Dtest', configfile="configs/EX4.json")
bnn.load(**dict)
# bnn.add_negative_constraint((-1.0, 1.0), [cx0, cx1, cy0, cy1])
# bnn.infer()
bnn.config["prior_type"] = "gaussian"
bnn.infer()
print(bnn.train_rmse())
# bnn.all_particles = bnn.all_particles[::-1]
# bnn.plot_pp(plot_title="Example 4 (Negative Constraint)", domain=np.arange(-0.05, 1.05, 0.01), ylims=(-3, 3), action='show', addons=addons)

X = np.linspace(-5.12, 5.12, 100)
Y = np.linspace(-5.12, 5.12, 100)
xx, yy = np.meshgrid(X, Y)

X_test = torch.from_numpy(np.vstack([xx.ravel(), yy.ravel()]).T).float()
Y_mean, Y_std = bnn.predict_all(X_test/5.12)

Y_test = rastrigin(X_test, 0.0)

plt.contourf(xx, yy, Y_mean.reshape(100,100), 20, cmap='viridis_r')
plt.colorbar()

plt.figure()
plt.contourf(xx, yy, Y_std.reshape(100,100), 20, cmap='jet')
plt.colorbar()
plt.plot(x[:, 0], x[:, 1], 'ro')

plt.figure()
plt.contourf(xx, yy, -Y_std.reshape(100,100) + Y_mean.reshape(100,100), 20, cmap='viridis_r')
plt.colorbar()
plt.contour(xx, yy,  -Y_std.reshape(100,100) + Y_mean.reshape(100,100), levels=[min(Y_mean)], colors='k', linestyles='dashed')
plt.plot(x[:, 0], x[:, 1], 'ro')

plt.figure()
plt.contourf(xx, yy, Y_test.reshape(100,100), 20, cmap='viridis_r')
plt.colorbar()
plt.show()



