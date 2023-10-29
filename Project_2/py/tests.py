from myloads import *
from mlp import ReLU

# return preprocessed train and val datasets
(X_train, y_train), (X_val, y_val) = get_data()

# set architecture for the neural network 

regularization_param = 0.0007
optim_method_dict_sgd = {"name": "SGD", "learning_rate": 0.1}
optim_method_dict_adagrad = {"name": "Adagrad", "learning_rate": 0.1}
optim_method_dict_rms = {"name": "RMSProp", "learning_rate": 0.001, "gamma": 0.95}
optim_method_dict_adam = {"name": "ADAM", "learning_rate": 0.0008, "gamma_adaptative": 0.98, "gamma_momentum": 0.5}
optim_method_dict = [optim_method_dict_sgd, optim_method_dict_adagrad, optim_method_dict_rms, optim_method_dict_adam]

# training of the neural network
batchsize = 32
n_epochs = 10

for optim_method in optim_method_dict:
    acc_log = []
    prec_log = []
    recall_log = []
    f1_log = []
    print(optim_method)
    network = []
    network.append(Dense(X_train.shape[1], 100, reg_param=regularization_param, **optim_method))
    network.append(ReLU())
    network.append(Dense(100, 200, reg_param=regularization_param, **optim_method))
    network.append(ReLU())
    network.append(Dense(200, 10, reg_param=regularization_param, **optim_method))

    for epoch in range(n_epochs):

        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize_=batchsize, shuffle_=True):
            train(network, x_batch, y_batch)
        acc_log.append(accuracy_score(predict(network, X_train), y_train))
        prec_log.append(precision_score(predict(network, X_train), y_train, average='weighted'))
        recall_log.append(recall_score(predict(network, X_train), y_train, average='weighted'))
        f1_log.append(f1_score(predict(network, X_train), y_train, average='weighted'))

        print("Epoch", epoch)
        print("Train accuracy:", acc_log[-1])
        print("Train precision:", prec_log[-1])
        print("Train recall:", recall_log[-1])
        print("Train f1-score:", f1_log[-1])

    plt.plot(acc_log)
    plt.legend(['SGD', 'Adagrad', 'RMSProp', 'ADAM'])
plt.grid()
plt.show()



