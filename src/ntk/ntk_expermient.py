from flax import linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import jax.flatten_util


class MLP(nn.Module):
    width: int

    def setup(self):
        self.dense1 = nn.Dense(features=self.width)
        self.dense2 = nn.Dense(features=self.width)
        self.dense3 = nn.Dense(features=self.width)
        self.dense4 = nn.Dense(features=1)

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        x = jax.nn.relu(x)
        x = self.dense3(x)
        x = jax.nn.relu(x)
        x = self.dense4(x)
        return x


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1000)
parser.add_argument("--epoch", type=int, default=1000)
args = parser.parse_args()
import datetime

now_time = datetime.datetime.now()
dir = "images/" + now_time.strftime("%Y/%m/%d/%H%M%S")
# os.mkdir(dir)
import os

os.makedirs(dir, exist_ok=True)


loss_dynamics_list = []

fig_loss = plt.figure()
fig_params_norm = plt.figure()

norm_list = []
loss_list = []
param_list = []
x_train = jnp.array([[-3], [0.5], [1], [3]])
y_train = jnp.array([2, -1.0, 1.0, -2.0])
x = jnp.linspace(-5, 5, 100)
x = jnp.expand_dims(x, axis=-1)
y = []

for width in [10, 100, 1000]:
    params_norm_list = []
    loss_dynamics = []
    key = jax.random.PRNGKey(0)

    model = MLP(width=width)
    params = model.init(key, x_train[0])
    init_params = params

    def loss_fn(params, x, y):
        y_pred = jax.vmap(model.apply, in_axes=(None, 0))(params, x)
        return jnp.mean((y_pred.reshape(-1) - y) ** 2)

    for i in range(args.epoch):
        grad_fn = jax.grad(loss_fn, argnums=0)
        grad = grad_fn(params, x_train, y_train)
        params = jax.tree_util.tree_map(lambda p, g: p - 0.001 * g, params, grad)
        if i % 5 == 0:
            print(loss_fn(params, x_train, y_train))
            loss_dynamics.append(loss_fn(params, x_train, y_train))
            params_norm = (
                jnp.linalg.norm(
                    jax.flatten_util.ravel_pytree(params)[0]
                    - jax.flatten_util.ravel_pytree(init_params)[0]
                )
            ) / jnp.linalg.norm(jax.flatten_util.ravel_pytree(init_params)[0])
            params_norm_list.append(params_norm)
    loss_dynamics_list.append(loss_dynamics)
    norm_list.append(params_norm_list)
    loss_list.append(loss_dynamics)
    param_list.append(params)
    y.append(jax.vmap(model.apply, in_axes=(None, 0))(params, x))

epoch_list = [i * 5 for i in range(args.epoch // 5)]
plt.plot(epoch_list, loss_list[0], label="width=10")
plt.plot(epoch_list, loss_list[1], label="width=100")
plt.plot(epoch_list, loss_list[2], label="width=1000")
plt.xlabel("epoch(t)")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.title("Training loss")
plt.legend()
plt.savefig(f"{dir}/loss_epoch_{args.epoch}.png")
plt.show()

plt.plot(epoch_list, norm_list[0], label="width=10")
plt.plot(epoch_list, norm_list[1], label="width=100")
plt.plot(epoch_list, norm_list[2], label="width=1000")
plt.xlabel("epoch(t)")
plt.ylabel("$\\frac{\|w(t) - w(0)\|}{\|w(0)\|}$")
plt.title("Parameter change")
plt.legend()
plt.savefig(f"{dir}/param_norm_epoch_{args.epoch}.png")
plt.show()


plt.scatter(x_train, y_train, label="train data")
plt.plot(x, y[0], label="width=10")
plt.plot(x, y[1], label="width=100")
plt.plot(x, y[2], label="width=1000")
plt.ylim(-5, 5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function approximation")
plt.legend()
plt.savefig(f"{dir}/function_approx_epoch_{args.epoch}.png")
plt.show()
