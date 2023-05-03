import numpy as np
import sympy as sp


def costo_individual_punto(x, y, w):
    y_pred = w * x
    cost = (y_pred - y) ** 2
    return cost


def grad(x, y, w):
    grad = 2 * x * ((w * x) - y)
    return grad


def decenso_gradiente(gradiente, x_0, alpha, n_iteraciones=50, error=1e-06):
    vector = x_0
    vector_history = []
    gradient_history = []
    for _ in range(n_iteraciones):
        vector_history.append(vector)
        grad = gradiente(vector)
        diff = -alpha * grad
        gradient_history.append(np.abs(alpha * grad))
        if np.all(np.abs(diff) <= error):
            break
        vector += diff
    return vector, vector_history, gradient_history


def sdg(x, y, w, gamma=0.001, iteraciones=60):
    x = x
    y = y
    all_costs = []
    w = w
    gamma = gamma
    iteraciones = iteraciones

    for k in range(iteraciones):
        for i in range(len(x)):
            cost = costo_individual_punto(x[i], y[i], w)
            grad_w = grad(x[i], y[i], w)
            w = w - (gamma * grad_w)  # in sgd update takes place after every point
        all_costs.append(cost)
    return w, all_costs


def batch(x, y, w, gamma=0.001, iteraciones=60):
    x = x
    y = y
    gamma = gamma
    w = w
    iteraciones = iteraciones
    all_costs = []

    for k in range(iteraciones):
        cost = 0
        for i in range(len(x)):
            a = costo_individual_punto(x[i], y[i], w)
            cost += a
        cost_med = cost / len(x)
        all_costs.append(cost_med)
        grad_w = 0
        for j in range(len(x)):
            b = grad(x[j], y[j], w)
            grad_w += b
        grad_w_med = grad_w / len(x)
        w = w - (gamma * grad_w_med)
    return w, all_costs


def mini_batch(x, y, w, gamma=0.001, batch_size=5, iteraciones=60):
    x = x
    y = y
    w = w
    gamma = gamma
    batch_size = 5
    all_costs = []
    iteraciones = 60
    for k in range(iteraciones):
        for j in range(int(len(x) / batch_size)):
            cost4 = 0
            for i in range(batch_size):
                z1 = costo_individual_punto(x[i], y[i], w)
                cost4 += z1
            if j == 1:
                all_costs.append(cost4 / batch_size)  # avearge cost of that batch
            grad_w41 = 0
            for n in range(batch_size):
                f1 = grad(x[i], y[i], w)
                grad_w41 += f1

            grad_w42 = grad_w41 / batch_size  # average grad of that function

            w = w - (gamma * grad_w42)  # update takes place after every batch
    return w, all_costs


def sgd(
    gradiente,
    x,
    y,
    start,
    alpha=0.1,
    batch_size=1,
    n_iteraciones=50,
    error=1e-06,
    dtype="float64",
    random_state=None,
):
    dtype_ = np.dtype(dtype)
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)

    n_obs = x.shape[0]
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    vector = np.array(start, dtype=dtype_)

    learn_rate = np.array(alpha, dtype=dtype_)
    batch_size = int(batch_size)

    tolerance = np.array(error, dtype=dtype_)

    for _ in range(n_iteraciones):
        rng.shuffle(xy)

        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            grad = np.array(gradiente(x_batch))
            diff = -learn_rate * grad

            if np.all(np.abs(diff) <= tolerance):
                break

            vector += diff
    return vector if vector.shape else vector.item()
