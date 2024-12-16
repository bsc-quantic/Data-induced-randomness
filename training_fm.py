import numpy as np 
from classes import fm_cost_new 

# File for training the feature map classifier

file = f"data/Classifier_FM/Vs/V{2}.npz"
Vn = np.load(file)[np.load(file).files[0]]

n_train = 50
seed = [1,2,3,4,5]

n_qubits = [4]
layers =[2,4,6,8,10,12]

brick = False



for n in n_qubits: 
    for l in layers: 
        for s in seed:
            qn =  fm_cost_new(n_qubits=n,layers= l, V = Vn, training_size=n_train, seed= s, data_reversed = False, brick = brick)
            

            _,params, cost_values, thetas = qn.minimize_funct(method='L-BFGS-B')
            if brick:
                directory = 'Params_trained_brick'
            else: 
                directory = 'Params_trained_non_brick'
            file_train_x =  f"data/Classifier_FM/{directory}/Train/training_set_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
            file_train_y =  f"data/Classifier_FM/{directory}/Train/training_labels_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
            file_params = f"data/Classifier_FM/{directory}/nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
            file_cost = f"data/Classifier_FM/{directory}/Cost/cost_values_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
            file_angles = f"data/Classifier_FM/{directory}/Cost/angles_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"

            np.savez(file_train_x, qn.train_data)
            np.savez(file_train_y, qn.labels)
            np.savez(file_params, params)
            np.savez(file_cost, cost_values)
            np.savez(file_angles, thetas )

