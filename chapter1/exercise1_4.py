import matplotlib.pyplot as plt
import numpy as np


def evaluate_h(w, X):
    '''
    Evaluate the hypothesis, h \in H, which is described by its set of weights,
    w, against 1 or more points in the input space (stored in a matrix).
    '''
    assert len(w.shape) == 1
    assert len(X.shape) == 2
    assert w.shape[0] == X.shape[0]

    return np.sign(w @ X)


def run_perceptron(w_initial, X_training, y_training, iteration_callback=None):
    w = w_initial.copy()
    n = 0
    while True:
        y = evaluate_h(w, X_training)

        if iteration_callback:
            iteration_callback(n, w)

        correct = y == y_training

        if np.all(correct):
            return w
        else:
            i = np.argmax(~correct)  # indice of first misclassified point
            w = w + y_training[i]*X_training[:, i]
        n = n + 1


def plot_hypothesis(x_coordinates, w, *plot_args, **plot_kwargs):
    m = -w[1]/w[2] if w[2] != 0 else 0
    b = -w[0]/w[2] if w[2] != 0 else 0
    y_coordinates = m*x_coordinates + b
    plt.plot(x_coordinates, y_coordinates, *plot_args, **plot_kwargs)


w_actual = np.array([1, 1, 1])

w_initial = np.array([3, -50, 0])

num_dimensions = 2

num_training_samples = 20

training_data_range = 10

X_training = np.vstack([
    np.ones(num_training_samples),
    np.random.uniform(
        -training_data_range,
        training_data_range,
        (num_dimensions, num_training_samples),
    ),
])

y_training = evaluate_h(w_actual, X_training)

x_coordinates = X_training[1, :]
y_coordinates = X_training[2, :]
colors = ['r' if y > 0 else 'b' for y in y_training]
plt.scatter(x_coordinates, y_coordinates, c=colors)

x_coordinates_hypothesis = np.array([-training_data_range, training_data_range])

def plot_iteration(n, w):
    if n % 5 == 0:
        label = 'iteration {}'.format(n)
        plot_hypothesis(x_coordinates_hypothesis, w, 'k:', label=label)

w_final = run_perceptron(w_initial, X_training, y_training, plot_iteration)

plot_hypothesis(x_coordinates_hypothesis, w_final, 'k', label='final')
plot_hypothesis(x_coordinates_hypothesis, w_actual, 'y', label='actual')

plt.legend()
plt.xlim(-training_data_range, training_data_range)
plt.ylim(-training_data_range, training_data_range)
plt.show()
