from myloads import *


def forward(network, X):
    activations = []

    temp_input = X
    for layer in network:
        temp_activation = layer.forward(temp_input)
        activations.append(temp_activation)
        temp_input = temp_activation

    assert len(activations) == len(network)
    return activations


def predict(network, X):
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y):
    # Get the layer activations
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    # Propagate gradients through the network>
    grad_input = loss_grad
    for i, layer in enumerate(reversed(network)):
        grad_input = layer.backward(layer_inputs[len(network) - i - 1], grad_input)

    return np.mean(loss)
