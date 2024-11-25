import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.hidden_dim = hidden_dim  # store hidden_dim as an attribute
        self.activation_fn = activation  # activation function
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize weights more conservatively
        self.weights1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.weights2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)

        self.bias1 = np.zeros((1, hidden_dim))  # bias for layer 1
        self.bias2 = np.zeros((1, output_dim))  # bias for layer 2

        self.hidden_layer = None
        self.hidden_layer_activation = None
        self.output_layer = None

        self.gradients = {}


    def forward(self, X):
        # forward pass, apply layers to input X
        self.hidden_layer = np.dot(X, self.weights1) + self.bias1
        if self.activation_fn == 'tanh':
            self.hidden_layer_activation = np.tanh(self.hidden_layer)
        elif self.activation_fn == 'relu':
            self.hidden_layer_activation = np.maximum(self.hidden_layer, 0)
        elif self.activation_fn == 'sigmoid':
            self.hidden_layer_activation = 1 / (1 + np.exp(-self.hidden_layer))
        else:
            raise ValueError("Invalid activation function")
        self.output_layer = np.dot(self.hidden_layer_activation, self.weights2) + self.bias2
        out = self.output_layer
        return out

    def backward(self, X, y):
        m = X.shape[0]
        d_output = self.output_layer - y
        d_weights2 = (1 / m) * np.dot(self.hidden_layer_activation.T, d_output)
        d_bias2 = np.sum(d_output, axis=0, keepdims=True)
        d_hidden_layer_activation = np.dot(d_output, self.weights2.T)
        if self.activation_fn == 'tanh':
            d_hidden_layer = d_hidden_layer_activation * (1 - np.tanh(self.hidden_layer) ** 2)
        elif self.activation_fn == 'relu':
            d_hidden_layer = d_hidden_layer_activation * (self.hidden_layer > 0)
        elif self.activation_fn == 'sigmoid':
            sig_hidden_layer = 1 / (1 + np.exp(-self.hidden_layer))
            d_hidden_layer = d_hidden_layer_activation * sig_hidden_layer * (1 - sig_hidden_layer)
        else:
            raise ValueError("Invalid activation function")

        d_weights1 = (1 / m) * np.dot(X.T, d_hidden_layer)
        d_bias1 = (1 / m) * np.sum(d_hidden_layer, axis=0, keepdims=True)

        # Clipping gradients to avoid exploding gradients
        np.clip(d_weights1, -1, 1, out=d_weights1)
        np.clip(d_weights2, -1, 1, out=d_weights2)
        np.clip(d_bias1, -1, 1, out=d_bias1)
        np.clip(d_bias2, -1, 1, out=d_bias2)

        # update weights with gradient descent
        self.weights1 -= self.lr * d_weights1
        self.weights2 -= self.lr * d_weights2
        self.bias1 -= self.lr * d_bias1
        self.bias2 -= self.lr * d_bias2

        # store gradients for visualization
        self.gradients = {
            'd_weights1': d_weights1,
            'd_weights2': d_weights2,
            'd_bias1': d_bias1,
            'd_bias2': d_bias2
        }


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    ax_hidden.scatter(
        *mlp.hidden_layer_activation[:, :3].T,
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7
    )

    # Plot the input space
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    hidden = np.sign(mlp.forward(grid))
    hidden = hidden.reshape(xx.shape)
    ax_input.contourf(
        xx,
        yy,
        hidden,
        cmap='bwr',
        alpha=0.5
    )
    ax_input.scatter(
        X[:, 0],
        X[:, 1],
        c=y.ravel(),
        cmap='bwr',
        edgecolors='k',
        alpha=0.7
    )

    ax_input.set_title('Input Space at Step {}'.format(frame * 10))
    ax_hidden.set_title('Hidden Space at Step {}'.format(frame * 10))


    ax_gradient.set_title('Gradients at Step {}'.format(frame * 10))
    input_positions = np.linspace(-1, 1, mlp.input_dim)
    hidden_positions = np.linspace(-1, 1, mlp.hidden_dim)
    output_positions = [0]

    for i, input_position in enumerate(input_positions):
        for j, hidden_position in enumerate(hidden_positions):
            gradient_magnitude = abs(mlp.gradients['d_weights1'][i, j])
            ax_gradient.plot(
                [0, 1],
                [input_position, hidden_position],
                color='purple',
                linewidth=gradient_magnitude * 10,
                alpha=0.7
            )

    for j, hidden_position in enumerate(hidden_positions):
        for k, output_position in enumerate(output_positions):
            gradient_magnitude = abs(mlp.gradients['d_weights2'][j, k])
            ax_gradient.plot(
                [1, 2],
                [hidden_position, output_position],
                color='purple',
                linewidth=gradient_magnitude * 10,
                alpha=0.7
            )

    for i, input_position in enumerate(input_positions):
        ax_gradient.add_patch(
            Circle(
                (0, input_position),
                0.1,
                color='blue',
                ec='purple',
                lw=2
            )
        )
    for j, hidden_position in enumerate(hidden_positions):
        ax_gradient.add_patch(
            Circle(
                (1, hidden_position),
                0.1,
                color='orange',
                ec='purple',
                lw=2
            )
        )
    for k, output_position in enumerate(output_positions):
        ax_gradient.add_patch(
            Circle(
                (2, output_position),
                0.1,
                color='red',
                ec='purple',
                lw=2
            )
        )




def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                                      ax_gradient=ax_gradient, X=X, y=y),
                        frames=step_num//10, repeat=False)

    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)