import numpy as np 
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z
from classes import classifier, moment_analytic


def moment_t_combined(t, qn, params, sample_list, y_label):
    t_moment = []
    f = SymbolicHamiltonian(np.prod([Z(i) for i in range(qn.qubits)]))
    total_samples = len(sample_list)
    # obs  = Hamiltonian(qn.n_qubits, matrix = f)
    for x,y in zip(sample_list, y_label):
        circ = qn.circuit_ansatz(x, params)
        exp_value = f.expectation(circ().state())
        My = (1+y*exp_value)/2
        t_moment.append(My**t)
    return np.mean(t_moment), np.std(t_moment, ddof=1)/np.sqrt(total_samples)

am = moment_analytic() 



# We define the parameters 
n_train = 50
Ms= {}
n = 8
epsilon = 0.07
ts= [i for i in range(1,5)]
layers = [2,4,6,8,10,12]
seed = [1,2,3,4,5]
eigenvalues = [1, 0]
multiplicities = [2**(n-1), 2**(n-1)]


M = []
for t in ts:
    M.append((2*t/epsilon)**2*(1+3/(4*2**(2*(n-1)))))
Ms[n] = M
moment_Haar = {t: am.compute_moments(eigenvalues,multiplicities, t) for t in ts}
M_t_train, error_t_train, M_t_test, error_t_test = {},{},{},{}
M_t_random = {}
error_t_random = {}

# We read the file with the unitary that later creates the dataset 
file = f"data/Classifier_FM/Vs/V{2}.npz"
Vn = np.load(file)[np.load(file).files[0]]

for i,t in enumerate(ts):
    m_l_train, m_l_test, error_l_train, error_l_test = [], [],[],[]
    m_l_random, error_l_random = [], []
    for l in layers:
        qn = classifier( n, l, V = Vn, training_size=int(Ms[n][i]), seed= 'random')
        m_s_train, error_s_train, m_s_test, error_s_test = [], [], [],[]
        for s in seed: 

            # We read the optimal parameters and the train data used in the optimization.  
            file_params = f"data/Classifier_circle/Params_trained/nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
            file_train_x =  f"data/Classifier_circle/Params_trained/Train/training_set_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
            file_train_y =  f"data/Classifier_circle/Params_trained/Train/training_labels_nqubits:{n}_layers:{l}-seed:{s}_n_train:{n_train}.npz"
            x_train = np.load(file_train_x)[np.load(file_train_x).files[0]]
            y_train = np.load(file_train_y)[np.load(file_train_y).files[0]]
            params_trained = np.load(file_params)[np.load(file_params).files[0]]
            # We create the test set. 
            x_test,y_test = qn.create_dataset(int(Ms[n][i]))
            # we compute the t-moments for the training and test set, together with its associated Monte Carlo error.  
            mean_plus_test, mc_error_test = moment_t_combined(t, qn, params_trained, x_test, y_test)
            mean_plus_train, mc_error_train = moment_t_combined(t, qn, params_trained, x_train, y_train)
            m_s_train.append(mean_plus_train)
            error_s_train.append(mc_error_train)
            m_s_test.append(mean_plus_test)
            error_s_test.append(mc_error_test)
        m_l_train.append(np.abs(np.mean(m_s_train)-moment_Haar[t]))
        error_l_train.append(np.sqrt(np.sum(error_s_train))/len(seed))
        m_l_test.append(np.abs(np.mean(m_s_test)-moment_Haar[t]))
        error_l_test.append(np.sqrt(np.sum(error_s_test))/len(seed))
        m_r, error_r = [], []
        # As the number of layers increase, the parameter space increases so we sample more parameters. 
        random_ite = 10*l
        for rand in range(random_ite):
            random_params = np.random.rand(l*n*4)*4*np.pi-2*np.pi
            mean_random, mc_error_random = moment_t_combined(t, qn, random_params, x_test, y_test)
            m_r.append(mean_random)
            error_r.append(mc_error_random)
        m_l_random.append(np.abs(np.mean(m_r)-moment_Haar[t]))
        error_l_random.append(np.sqrt(np.sum(error_r))/random_ite)

    M_t_train[t] = m_l_train
    error_t_train[t] = error_l_train
    M_t_test[t] = m_l_test
    error_t_test[t] = error_l_test
    M_t_random[t] = m_l_random
    error_t_random[t] = error_l_random

file_M = f"data/Classifier_circle/RvsL/Trained/MY_TRAIN_R_nqubits:{n}_t:{ts}-layers:{layers}-epsilon_{epsilon}_n_train:{n_train}.npz"
file_error =  f"data/Classifier_circle/RvsL/Trained/MY_TRAIN_Error_nqubits:{n}_t:{ts}-layers:{layers}-epsilon_{epsilon}_n_train:{n_train}.npz"
file_M_rand = f"data/Classifier_circle/RvsL/Random/MY_R_nqubits:{n}_t:{ts}-layers:{layers}-epsilon_{epsilon}_n_train:{n_train}.npz"
file_error_rand =  f"data/Classifier_circle/RvsL/Random/MY_Error_nqubits:{n}_t:{ts}-layers:{layers}-epsilon_{epsilon}_n_train:{n_train}.npz"
file_M_test = f"data/Classifier_circle/RvsL/Trained/MY_TEST_R_nqubits:{n}_t:{ts}-layers:{layers}-epsilon_{epsilon}_n_train:{n_train}.npz"
file_error_test =  f"data/Classifier_circle/RvsL/Trained/MY_TEST_Error_nqubits:{n}_t:{ts}-layers:{layers}-epsilon_{epsilon}_n_train:{n_train}.npz"
np.savez(file_M,  M_t_train)
np.savez(file_error,  error_t_train)
np.savez(file_M_test,  M_t_test)
np.savez(file_error_test,  error_t_test)
np.savez(file_M_rand,  M_t_random)
np.savez(file_error_rand,  error_t_random)