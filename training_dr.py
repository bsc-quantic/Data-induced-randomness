import numpy as np
from classes import classifier 

# File for training the data re-uploading classifier. 


seed = [1,2,3,4,5]
n_train = 50
n_qubits = [2,4]
layers = [2,4,6,8,10,12]
weights = True
for n in n_qubits: 
    file = f"data/Classifier_FM/Vs/V{2}.npz"
    Vn = np.load(file)[np.load(file).files[0]]
    for l in layers: 
        for s in seed:

            qn =  classifier(qubits=n,layers= l, V = Vn, training_size=n_train, seed= s, weights = weights)

            _,params, steps, cost_values, thetas = qn.minimize_funct(method='L-BFGS-B',
                                        options={'disp': True}, compile= True)
            
            if weights == True: 
                file_train_x =  f"data/Classifier_circle/Params_trained/Train/training_set_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
                file_train_y =  f"data/Classifier_circle/Params_trained/Train/training_labels_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
                file_params = f"data/Classifier_circle/Params_trained/nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
                file_cost = f"data/Classifier_circle/Params_trained/Cost/cost_values_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"

                file_angles = f"data/Classifier_circle/Params_trained/Cost/angles_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
            np.savez(file_train_x, qn.training_set[0])
            np.savez(file_train_y, qn.training_set[1])
            np.savez(file_params, params)
            np.savez(file_cost, cost_values)
            np.savez(file_angles, thetas )


