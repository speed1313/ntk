from flax import linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import jax.flatten_util
import argparse
import datetime
import os
import csv



class MLP(nn.Module):
    width: int
    depth: int

    def setup(self):
        for i in range(self.depth):
            setattr(self, f"dense{i}", nn.Dense(features=self.width))
        self.dense = nn.Dense(features=1)

    def __call__(self, x):
        for i in range(self.depth):
            x = getattr(self, f"dense{i}")(x)
            x = jax.nn.relu(x)
        x = self.dense(x)
        return x


def split_into_batches_random(arr, batch_size, rng_key):
    rng_key, subkey = jax.random.split(rng_key)
    shuffled_indices = jax.random.permutation(subkey, jnp.arange(arr.shape[0]))

    arr_shuffled = arr[shuffled_indices]
    num_batches = arr.shape[0] // batch_size
    batches = [
        arr_shuffled[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
    ]

    if num_batches * batch_size < arr.shape[0]:
        remaining_batch = arr_shuffled[num_batches * batch_size :]
        batches.append(remaining_batch)
    return batches, rng_key


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--mode", type=str, default="minibatch")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--depth", type=int, default=3)
args = parser.parse_args()

mode = args.mode
batch_size = args.batch_size
lr = args.lr
widths = [1000, 100, 10]  # , 2000]
num_width = len(widths)

now_time = datetime.datetime.now()
dir = "images/" + now_time.strftime("%Y/%m/%d/%H%M%S")
os.makedirs(dir, exist_ok=True)

# output config file
# first row keys
# second row values
with open(f"{dir}/config.csv", "w") as f:
    writer = csv.writer(f)
    first_row = []
    for key in vars(args).keys():
        first_row.append(key)
    writer.writerow(first_row)
    second_row = []
    for value in vars(args).values():
        second_row.append(value)
    writer.writerow(second_row)


loss_dynamics_list = []
norm_list = []
loss_list = []
ntk_list_per_width = [] * num_width

def true_fn(x):
    return x**3 / 20
    # return jnp.sin(x)

# train data
train_num = 10
x_train = jnp.linspace(-5, 5, train_num)
x_train = jnp.expand_dims(x_train, axis=-1)
y_train = jax.vmap(true_fn)(x_train).reshape(-1)

# plot data
x = jnp.linspace(-10, 10, 100)
x = jnp.expand_dims(x, axis=-1)
y = []


plt.scatter(x_train, y_train, label="train data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Train data")
plt.legend()
plt.savefig(f"{dir}/train_data.png")
plt.show()




for width in widths:
    params_norm_list = []
    loss_dynamics = []
    ntk_list = []
    key = jax.random.PRNGKey(0)
    model = MLP(width=width, depth=args.depth)
    params = model.init(key, x_train[0])
    init_params = params

    @jax.jit
    def loss_fn(params, x, y):
        y_pred = jax.vmap(model.apply, in_axes=(None, 0))(params, x)
        return jnp.mean((y_pred.reshape(-1) - y) ** 2)

    # this function is defined to benefit from jit.
    # model's input shape is defined after mode.init
    @jax.jit
    def NTK(params, x, y):
        """
        J_fn = jax.jacrev(model.apply, argnums=0)
        J_x = J_fn(params, x)
        grad_x = jax.flatten_util.ravel_pytree(J_x)[0]
        J_y = J_fn(params, y)
        grad_y = jax.flatten_util.ravel_pytree(J_y)[0]
        ntk = grad_x @ grad_y.T
        return ntk
        """

        def f(model, params, x):
            return model.apply(params, x)[0]

        grad_x = jax.grad(f, argnums=1)(model, params, x)
        grad_x = jax.flatten_util.ravel_pytree(grad_x)[0]
        # grad_y = jax.grad(f, argnums=1)(model, params, y)
        # grad_y = jax.flatten_util.ravel_pytree(grad_y)[0]
        ntk = grad_x @ grad_x.T
        return ntk

    ntk_init = NTK(params, x_train[0], x_train[1])
    print("ntk_init: ", ntk_init)
    for i in range(args.epoch):
        if mode == "full":
            loss, grad = jax.value_and_grad(loss_fn, argnums=0)(
                params, x_train, y_train
            )
            params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grad)
        elif mode == "sgd":
            key, subkey = jax.random.split(key)
            random_order = jax.random.permutation(subkey, jnp.arange(len(x_train)))
            for idx in random_order:
                batch_x = x_train[idx : idx + 1]
                batch_y = y_train[idx : idx + 1]
                loss, grad = jax.value_and_grad(loss_fn, argnums=0)(
                    params, batch_x, batch_y
                )
                params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grad)
        elif mode == "minibatch":
            idxs = jnp.arange(len(x_train))
            batches, key = split_into_batches_random(idxs, batch_size, key)

            for batch in batches:
                batch_x = x_train[batch]
                batch_y = y_train[batch]
                loss, grad = jax.value_and_grad(loss_fn, argnums=0)(
                    params, batch_x, batch_y
                )
                params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grad)
        else:
            raise ValueError("mode must be 'full', 'sgd' or 'minibatch'")

        if i % 5 == 0:
            loss = loss_fn(params, x_train, y_train)
            print(loss)
            loss_dynamics.append(loss)
            params_norm = (
                jnp.linalg.norm(
                    jax.flatten_util.ravel_pytree(params)[0]
                    - jax.flatten_util.ravel_pytree(init_params)[0]
                )
            ) / jnp.linalg.norm(jax.flatten_util.ravel_pytree(init_params)[0])
            params_norm_list.append(params_norm)
        if i % 5 == 0:
            ntk = NTK(params, x_train[0], x_train[1])
            ntk_list.append(jnp.linalg.norm(ntk - ntk_init) / jnp.linalg.norm(ntk_init))

    loss_dynamics_list.append(loss_dynamics)
    norm_list.append(params_norm_list)
    loss_list.append(loss_dynamics)
    y.append(jax.vmap(model.apply, in_axes=(None, 0))(params, x))

    ntk_list_per_width.append(ntk_list)

epoch_list = [i * 5 for i in range(args.epoch // 5)]
for i in range(num_width):
    plt.plot(epoch_list, loss_list[i], label=f"width={widths[i]}")
plt.xlabel("epoch(t)")
plt.ylabel("loss")
plt.ylim(0, 5)
plt.xscale("log")
plt.title("Training loss")
plt.legend()
plt.savefig(f"{dir}/loss_epoch_{args.epoch}.png")
plt.show()
for i in range(num_width):
    plt.plot(epoch_list, norm_list[i], label=f"width={widths[i]}")

plt.xlabel("epoch(t)")
plt.ylabel("$\\frac{\|w(t) - w(0)\|}{\|w(0)\|}$")
plt.title("Parameter change")
plt.legend()
plt.xscale("log")
plt.savefig(f"{dir}/param_norm_epoch_{args.epoch}.png")
plt.show()


plt.scatter(x_train, y_train, label="train data")
for i in range(num_width):
    plt.plot(x, y[i], label=f"width={widths[i]}")

plt.xlim(-10, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function approximation")
plt.legend()
plt.savefig(f"{dir}/function_approx_epoch_{args.epoch}.png")
plt.show()

epoch_list = [i * 5 for i in range(args.epoch // 5)]
for i in range(num_width):
    plt.plot(epoch_list, ntk_list_per_width[i], label=f"width={widths[i]}")
plt.xlabel("epoch(t)")
plt.ylabel("$\\frac{\|NTK(t) - NTK(0)\|}{\|NTK(0)\|}$")
plt.title("$NTK(x_{train[0]}, x_{train[0]})$")
plt.legend()
plt.xscale("log")
plt.savefig(f"{dir}/ntk_epoch_{args.epoch}.png")
plt.show()
