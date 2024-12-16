import numpy as np 
# Qibo modules
from qibo.models import Circuit
from qibo import gates
from qibo.optimizers import cmaes

#optimization method
from scipy.optimize import minimize
from qibo.hamiltonians import Hamiltonian

#Ploting results
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from matplotlib.colors import Normalize
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z

from scipy.special import binom, loggamma



class classifier:
    def __init__(self, qubits, layers, V, training_size=100, data_size = 5000, seed=7, weights = True):

        """"Class that includes all the necessary for the training of the data re-uploading classifier

        Args: 
            - qubits (int): Number of qubits that the classifier has
            - layers(int): Number of layers that our circuit ansatz has
            - V (array): unitary matrix that creates the data set. 
            - training size (int): number of training points
            - data_size (int): data for visualization
            - seed (int): Random seed
            - weights (bool): if the model includes trainable weights or not
            """
        if seed == 'random': 
            np.random.seed(np.random.randint(100, size=1))
        else: 
            np.random.seed(seed)
        self.V = V
        self.qubits = qubits
        self.data_size = data_size
        self.layers = layers 
        self.training_size = training_size
        self.training_set= self.create_mesh(int(np.sqrt(self.training_size)))
        self.params = np.random.rand(self.layers*self.qubits*4)
        self.f = SymbolicHamiltonian(np.prod([Z(i) for i in range(self.qubits)]))
        self.weights = weights

    # Circuit to create the data-set
    def featuremapZZ(self, x, n_qubits = 2):
        C = Circuit(n_qubits)
        pi = np.pi
        data = np.array(list(x)*int(n_qubits/2))
        for n in range(n_qubits):
            C.add(gates.H(n))
            C.add(gates.U1(n, theta = 2*data[n], trainable= False))

        for n in range(n_qubits):
            for n_ in range(n):
                C.add(gates.CNOT(n_,n))
                C.add(gates.U1(n, theta = 2*(pi-data[n_])*(pi-data[n]), trainable=False))
                C.add(gates.CNOT(n_,n))
        return C

    # For visualization
    def create_mesh(self, data_size, start = 0, end = 2*np.pi): 
        start = 0
        end = 2 * np.pi
        spacing = end/data_size

        # Generate the points in each dimension
        x1 = np.arange(start, end + spacing, spacing)
        x2 = np.arange(start, end + spacing, spacing)
        # Create the meshgrid
        data = np.array([[i, j] for i in x1 for j in x2])
        f = SymbolicHamiltonian(np.prod([Z(i) for i in range(2)]))
        V = self.V              
        obs = np.conjugate(V.T)@(f@V)
        obs  = Hamiltonian(2, matrix = obs)
        y = []
        for _,x_1 in enumerate(x1):
            for j, x_2 in enumerate(x2):
                fm = self.featuremapZZ([x_1, x_2], 2)
                fm = fm+fm
                exp_value = obs.expectation(fm().state())
                if exp_value >= 0:
                    y.append(1)
                elif exp_value < 0:
                    y.append(-1) 
        return data, y 


    
    def create_dataset(self, data_size):
        data = [np.random.uniform(0,2*np.pi, size = 2) for i in range(data_size)]
        f = SymbolicHamiltonian(np.prod([Z(i) for i in range(2)]))
        V = self.V              
        obs = np.conjugate(V.T)@(f@V)
        obs  = Hamiltonian(2, matrix = obs)
        y = []
        for x_train in data:
            fm = self.featuremapZZ(x_train, 2)
            fm = fm+fm
            exp_value = obs.expectation(fm().state())
            if exp_value >= 0:
                y.append(1)
            elif exp_value < 0:
                y.append(-1)

        return data, y


    def circuit_ansatz(self,x, params = None): 
        if params is None: 
            params = self.params
        else:
            params =  params.reshape(self.layers, self.qubits, 4)
        C = Circuit(self.qubits)
        for p in params: 
            for idx, p_i in enumerate(p):
                if self.weights == True: 
                    C.add(gates.U3(idx, theta= p_i[0]*x[0] + p_i[1], phi= p_i[2]*x[1]+ p_i[3], lam=0))
                else: 
                    C.add(gates.U3(idx, theta= x[0] + p_i[1], phi= x[1]+ p_i[3], lam=0, trainable = True))
                #We apply a Cnot if we have the two-qubit ansatz
                if idx<self.qubits-1:
                    C.add(gates.CNOT(idx,idx+1))
                elif idx == self.qubits-1:
                    C.add(gates.CNOT(idx, 0))

        return C

    def cost_function(self, params = None, x_t = None, y_t = None ): 
        """This function computes the normalized sum of the fidelities between the label state for the classification and the state obtained
        via passing the data through the circuit (with the given params). 

        Args: 
            -params: The params will be updated after each step of the minimization method. 
        Returns: 
        - Value of the cost function. 
        """
        
        if x_t is None: 
            x_t, y_t  = self.training_set
        #If the arg of the function is given, we upload the value of the params of the class
        if params is not None: 
            self.params = params.reshape(self.layers, self.qubits, 4)

    
        cf = 0 
        obs = self.f
        for x, y in zip(x_t, y_t):
            C = self.circuit_ansatz(x)
            exp_value = obs.expectation(C().state())
            py = 1/2*(1+y*exp_value)
            #We add to the cost function the contribution of each data-point        
            cf -=py
        cf /= len(y_t)
        return cf 



    def minimize_funct(self,method='l-bfgs', options=None):
        """"This function minimizes the cost function in the space of the parameters. Then 
        it returns the value of the function when the optimization has finished and the 
        values of the parameters that acomplish the desired optimization. Also computes 
        the values of the gradients of the parameters.
        
        Args: 
            - Method (str): Method used to minimize the cost function. Options: ['bfgs', 
                'L-BFGS-B', 'Nelder-Mead', ...]
            - Options (dict): Options for the minimization
        Returns: 
        
            -result (float): value of the cost function after the minimization. 
            -params (list): list of the parameters that accomplish the minimization.
            - steps (list): list of the steps done by the algorithm
            - cost_values (list): value of the cost function at each step
            - thetas (list): optimized angles after each iteration
              """

        steps = []
        cost_values = []
        thetas = []
        def save_step(k):
            cost_values.append(self.cost_function(k))
            steps.append(k)
            thetas.append(self.params)

        m = minimize(self.cost_function, self.params,
                    method = method, options=options, callback= save_step)

       
        result = m.fun
        params = m.x

        return result, params, steps, cost_values, thetas



    def eval_test_set_fidelity(self,params_opt, xandy = None):
        """This function returns the labels of the classification states predicted for each data-point. 
        We run our circuit for all the data with the optimized parameters

        Args: 
            -params_opt (list): Parameters used in the classifier circuit. 
            - xandy (array): in case we want to evaluate x and y that are different from the training set. 
        Returns: 
            - labels (list): predicted labels for each data-point."""
        if xandy is not None: 
            x_s= xandy[0]
            y = xandy[1]
        else: 
            x_s = self.training_set[0]
            y = self.training_set[1]
        self.params = params_opt
        labels = np.ones(len(y))
        for i,x in enumerate(x_s):
            C = self.circuit_ansatz(x)
            obs = self.f
            exp_value = obs.expectation(C().state())
            if exp_value >0:
                labels[i] = 1
            else: 
                labels[i] = -1
        return labels

    def accuracy_score(y_pred, y_real):

        """Computes the accuracy as the ratio of the number of right predicted labels and the number of total attempts. 
        
            Args: 
                -y_pred (array of ints): array with the predicted labels.
                -y_real (array of ints): array with the true labels.

            Returns: 
                - proportion of right predictions. 
                
                """
        score = y_pred == y_real
        return score.sum() / len(y_real)

class moment_analytic():
    """This class computes the moments of a given observable as if they were averaged over Haar-random states, that is: 
    - E_S[<psi|O|psi>^t], and S = {|\psi>} is a Haar-random set of states.
    The analytical formula can be found in: https://arxiv.org/abs/2404.16211"""
    def find_multinomial_elements(self, t,G):
        elements = []
        if G == 2:
            for k in range(t + 1):
                elements.append((k, t - k))

        if G > 2:
            for k in range(t + 1):
                elem = [e for e in self.find_multinomial_elements(t-k, G - 1)]
                for e in elem:
                    elements.append((k, *e))

        return elements

    def multinomial(self, elements):
        if len(elements) == 1:
            return 1
        return binom(sum(elements), elements[-1]) * self.multinomial(elements[:-1])

    def compute_moments(self, eigenvalues, multiplicities, t):
        assert len(eigenvalues) == len(multiplicities)
        G = len(eigenvalues)
        elements = self.find_multinomial_elements(t, G)

        result = 0
        # eig:  product of lambda^k_i
        # multinomial(elem): 
        for elem in elements:
            eig = 1
            for l, k in zip(eigenvalues, elem):
                eig *= l**k
            loggam = loggamma(sum(multiplicities)/2) - loggamma(sum(multiplicities)/2 + t)
            for m, k in zip(multiplicities, elem):
                loggam += loggamma(m/2 + k) - loggamma(m/2)
            r = self.multinomial(elem) * np.exp(loggam) * eig
            result += r


        return result


    def get_degeneracy(eigvals):
        eigvals = np.round(eigvals, 4)
        unique_eigvals = np.unique(eigvals)
        deg={}
        alphas = {}
        for v in unique_eigvals:
            deg[v] = len(np.where(eigvals == v)[0])
            alphas[v] = deg[v]/2#*(1+v)
        return deg, alphas, unique_eigvals


    def compute_est_t_moment(values, moment):
        mu_bar_t = sum(np.power(values, moment))/len(values)
        mu_bar_2t = sum(np.power(values, 2*moment))/len(values)
        var_mu_bar_t = mu_bar_2t - mu_bar_t**2
        return mu_bar_t, var_mu_bar_t


    def compute_randomness_bounds(mu_bar, mu_lower, mu_upper):
        rnd_lower = np.abs(mu_bar - mu_lower)
        rnd_upper = np.abs(mu_bar - mu_upper)
        return rnd_lower, rnd_upper



class fm_cost_new:
    """ This class contains the necessary code to train the feature-map classifier. 

        Args: 
        - n_qubits (int): Number of qubits that the classifier has 
        - layers(int): Number of layers that our circuit ansatz has 
        - V (array): unitary matrix that creates the data set. 
        - method (str): optimization algorithm
        - training size (int): number of training points
        - data_size (int): data for visualization
        - seed (int) or (str): we can specify the seed or ask for a random one.  
        - data_reverser (bool): If we want the order of the training data to be changed when uploaded to the model. 
        - brick (bool): True if we want the brick ansatz, False if we want the non-brick ansatz. 
        """
    def __init__(self, n_qubits, layers, V = None, method = 'L-BFGS-B', data_size = 5000, training_size = 20, seed = 'random', data_reversed = False, brick = False):
        if seed == 'random': 
            np.random.seed(np.random.randint(100, size=1))
        else: 
            np.random.seed(seed)
        self.brick = brick #True-> feature-map brick. False-> Feature map first neighbours
        self.data_reversed = data_reversed
        self.n_qubits = n_qubits 
        self.layers = layers 
        self.method = method 
        self.params =np.random.rand((self.layers)*self.n_qubits)
        self.data_size = data_size
        self.training_size = training_size
        self.V = V
        self.f = SymbolicHamiltonian(np.prod([Z(i) for i in range(self.n_qubits)]))
        self.train_data, self.labels = self.create_mesh(int(np.sqrt(self.training_size)))
       
        


    def featuremapZZ(self, x, n_qubits = None ):
        if n_qubits is None:
            n_qubits = self.n_qubits
        C = Circuit(n_qubits)
        pi = np.pi
        if self.data_reversed is False: 
            if n_qubits%4 ==0:
                data = np.array(list(list(x)+ list(reversed(x)))*int((n_qubits)/4))
            else: 
                data = np.array(list(list(x)+ list(reversed(x)))*int((n_qubits-2)/4)+list(x))
        else: 
            data = np.array(list(list(x))*int((n_qubits/2)))
        for n in range(n_qubits):
            C.add(gates.H(n))
            C.add(gates.U1(n, theta = 2*data[n], trainable= False))
        for n in range(n_qubits):
            for n_ in range(n):
                C.add(gates.CNOT(n_,n))
                C.add(gates.U1(n, theta = 2*(pi-data[n_])*(pi-data[n]), trainable=False))
                C.add(gates.CNOT(n_,n))
        return C


    def featuremap_brick(self, x, brick = False):
        n_qubits = self.n_qubits
        C = Circuit(n_qubits)
        pi = np.pi
        data = list(x)*int((n_qubits)/2)
        for n in range(n_qubits):
            C.add(gates.H(n))
            C.add(gates.U1(n, theta = 2*data[n], trainable= False))

        for n in range(0,n_qubits,2):
            C.add(gates.CNOT(n,(n+1)%n_qubits))
            C.add(gates.U1((n+1)%n_qubits, theta = 2*(pi-data[n])*(pi-data[n+1]), trainable=False))
            C.add(gates.CNOT(n,(n+1)%n_qubits))

        if brick:
            for n in range(n_qubits):
                C.add(gates.H(n))
                C.add(gates.U1(n, theta = 2*data[n], trainable= False))
            for n in range(1,n_qubits,2):
                C.add(gates.CNOT(n,(n+1)%n_qubits))
                C.add(gates.U1((n+1)%n_qubits, theta = 2*(pi-data[n])*(pi-data[(n+1)%n_qubits]), trainable=False))
                C.add(gates.CNOT(n,(n+1)%n_qubits))
            return C
        else:
            return C+C
    
    def variational_circuit(self, params= None):
        if params is None: 
            params = self.params
        else: 
            params = params.reshape(self.layers, self.n_qubits)
        n_qubits = self.n_qubits
        layers = self.layers
        C = Circuit(n_qubits)
        for l in range(layers): 
            for n in range(n_qubits):
                C.add(gates.RY(n, params[l][n], trainable= True))
            for n in range(n_qubits):
                if n != n_qubits-1:
                    C.add(gates.CZ(n, n+1))
                elif n_qubits>2: 
                    C.add(gates.CZ(n, 0))
        return C
    

    def create_mesh(self, data_size, start = 0, end = 2*np.pi): 
        start = 0
        end = 2 * np.pi
        spacing = end/data_size

        # Generate the points in each dimension
        x1 = np.arange(start, end + spacing, spacing)
        x2 = np.arange(start, end + spacing, spacing)
        # Create the meshgrid
        data = np.array([[i, j] for i in x1 for j in x2])
        f = SymbolicHamiltonian(np.prod([Z(i) for i in range(2)]))
        V = self.V              
        obs = np.conjugate(V.T)@(f@V)
        obs  = Hamiltonian(2, matrix = obs)
        y = []
        for _,x_1 in enumerate(x1):
            for _, x_2 in enumerate(x2):
                fm = self.featuremapZZ([x_1, x_2], 2)
                fm = fm+fm
                exp_value = obs.expectation(fm().state())
                if exp_value >= 0:
                    y.append(1)
                elif exp_value < 0:
                    y.append(-1) 
        return data, y 


    def create_dataset(self, data_size):
        data = [np.random.uniform(0,2*np.pi, size = 2) for i in range(data_size)]
        f = SymbolicHamiltonian(np.prod([Z(i) for i in range(2)]))
        V = self.V              
        obs = np.conjugate(V.T)@(f@V)
        obs  = Hamiltonian(2, matrix = obs)
        y = []
        clean_data = []
        for x_train in data:
            fm = self.featuremapZZ(x_train, 2)
            fm = fm+fm
            exp_value = obs.expectation(fm().state())
            if exp_value >= 0:
                y.append(1)
                clean_data.append(x_train)
            elif exp_value < 0:
                y.append(-1)
                clean_data.append(x_train)

        return clean_data, y
    

    def cost(self, params, train_data = None, labels=None): 
        if train_data is None:
            train_data = self.train_data
            labels = self.labels
        n_qubits = self.n_qubits 
        layers = self.layers 
        self.params= params
        params = params.reshape(layers,n_qubits)
        cost = 0
        obs = self.f
        for x, y in zip(train_data, labels):
            fm = self.featuremap_brick(x, self.brick)
            circ = fm+self.variational_circuit(params)
            exp_value = obs.expectation(circ().state())
            py = 1/2*(1+y*exp_value)
            cost -= py
        cost /= len(labels)
        return cost 



    def minimize_funct(self, method='L-BFGS-B'):
        """"This function minimizes the cost function in the space of the parameters. Then 
        it returns the value of the function when the optimization has finished and the 
        values of the parameters that acomplish the desired optimization. Also computes 
        the values of the gradients of the parameters.
        
        Args: 
            - Method (str): Method used to minimize the cost function. Options: ['bfgs', 
                'L-BFGS-B', 'Nelder-Mead', ...]
            - Options (dict): Options for the minimization
        Returns: 
        
            -result (float): value of the cost function after the minimization. 
            -params (list): list of the parameters that accomplish the minimization.
            - steps (list): list of the steps done by the algorithm
            - cost_values (list): value of the cost function at each step
            - thetas (list): optimized angles after each iteration
              """



        cost_values = []
        thetas = []
        def save_step(k):
            # global steps
            print(self.cost(k))
            cost_values.append(self.cost(k))
            thetas.append(self.params)

        
        m = minimize(self.cost, self.params, args = (), method= method, callback=save_step)
            
        result = m.fun
        params = m.x

        return result, params, cost_values, thetas


    

    



            





