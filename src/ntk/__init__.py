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
        self.dense3 = nn.Dense(features=1)

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        x = jax.nn.relu(x)
        x = self.dense3(x)
        return x


"""
for i in range(10):
    key = jax.random.PRNGKey(i)
    model = MLP()
    x = jnp.linspace(-5, 5, 100)
    x = jnp.expand_dims(x, axis=-1)
    print(x.shape)
    print(x[0].shape)
    params = model.init(key, x[0])
    print(params["params"]["dense1"]["kernel"].shape)
    y = jax.vmap(model.apply, in_axes=(None, 0))(params, x)
    plt.plot(x, y, label=f"model {i}")
plt.legend()
plt.show()
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1000)
parser.add_argument("--epoch", type=int, default=1000)
args = parser.parse_args()

loss_dynamics_list = []

# training loop
for i in range(10):
    params_norm_list = []
    loss_dynamics = []
    key = jax.random.PRNGKey(i)
    x_train = jnp.array([[-3], [0.1], [3]])
    y_train = jnp.array([2, 0.2, 2])
    model = MLP(width=args.width)
    params = model.init(key, x_train[0])
    init_params = params

    def loss_fn(params, x, y):
        y_pred = jax.vmap(model.apply, in_axes=(None, 0))(params, x)
        return jnp.mean((y_pred.reshape(-1) - y) ** 2)

    for i in range(args.epoch):
        grad_fn = jax.grad(loss_fn, argnums=0)
        grad = grad_fn(params, x_train, y_train)
        params = jax.tree_util.tree_map(lambda p, g: p - 0.001 * g, params, grad)
        if i % 10 == 0:
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

    x = jnp.linspace(-5, 5, 100)
    x = jnp.expand_dims(x, axis=-1)
    y = jax.vmap(model.apply, in_axes=(None, 0))(params, x)
    plt.plot(x, y)
    plt.ylim(-5, 5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x_train, y_train)


plt.savefig(f"width_{args.width}_epoch_{args.epoch}.png")
plt.show()

plt.plot(
    [i * 10 for i in range(args.epoch // 10)],
    jnp.array(loss_dynamics_list).mean(axis=0),
)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(f"width_{args.width}_epoch_{args.epoch}_loss.png")
plt.show()

plt.plot([i * 10 for i in range(args.epoch // 10)], params_norm_list)
plt.xlabel("epoch")
plt.ylabel("params norm")
plt.savefig(f"width_{args.width}_epoch_{args.epoch}_params_norm.png")

import os

os._exit(0)


def NTK(model, params, x, y):
    def f(params, x):
        return model.apply(params, x)[0]

    grad_fn = jax.grad(f, argnums=0)
    grad_x = grad_fn(params, x)
    grad_y = grad_fn(params, y)
    grad_x = jax.flatten_util.ravel_pytree(grad_x)[0]

    grad_y = jax.flatten_util.ravel_pytree(grad_y)[0]
    ntk = jnp.dot(grad_x, grad_y.T)
    return ntk


model = MLP()
x = jnp.linspace(-3, 3, 10)
x = jnp.expand_dims(x, axis=-1)

key = jax.random.PRNGKey(0)
params = model.init(key, x[0])
print(model.apply(params, x[0]))
ntk = NTK(model, params, x[0], x[1])
print(ntk.shape)
print(ntk)

ntk_list = jnp.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        ntk = NTK(model, params, x[i], x[j])
        ntk_list = ntk_list.at[i, j].set(ntk)

plt.imshow(ntk_list)
plt.colorbar()
plt.show()
