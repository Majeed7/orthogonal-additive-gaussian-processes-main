import numpy as np 

def create_rank(scores): 
	"""
	Compute rank of each feature based on weight.
	
	"""
	scores = abs(scores)
	n, d = scores.shape
	ranks = []
	for i, score in enumerate(scores):
		# Random permutation to avoid bias due to equal weights.
		idx = np.random.permutation(d) 
		permutated_weights = score[idx]  
		permutated_rank=(-permutated_weights).argsort().argsort()+1
		rank = permutated_rank[np.argsort(idx)]

		ranks.append(rank)

	return np.array(ranks)

def generate_data(n=500, d=10, datatype='XOR'):
    if datatype == 'XOR':
         return generate_XOR(n, d)
    
    elif datatype == 'nonlinear_additive':
         return generate_additive_labels(n, d)

    elif datatype == 'simple_interaction':
         return generate_simple_interactions(n, d)
    
    elif datatype == 'poly_sin':
         return generate_dataset_poly_sine(n, d)
    
    elif datatype == 'squared_exp':
         return generate_dataset_squared_exponentials(n, d)
    
    
def generate_X(n_samples=100, n_features=10):
    #return np.random.uniform(-1, 1, (n_samples, n_features))
    return np.random.randn(n_samples, n_features)

def generate_dataset_poly_sine(n_samples=100, n_features=10):
    X = generate_X(n_samples, n_features)
    
    def fn(X):
        f1, f2, f3 = X[:, 0], X[:, 1], X[:, 2]
        y = f1**2 - 0.5 * f2**2 + f3**3 + np.sin(2 * np.pi * f1) 

        logit = np.exp(y) 
        prob_1 = np.expand_dims(1 / (1+logit) ,1)

        return y
    
    return X, fn(X), fn, np.arange(0,3), 'Poly Sine'

def generate_dataset_squared_exponentials(n_samples=100, n_features=10):
    X = generate_X(n_samples, n_features)
    
    def fn(X):
        logit = np.exp(np.sum(X[:,:4]**2, axis = 1) - 4.0) 

        prob_1 = np.expand_dims(1 / (1+logit) ,1)
        prob_0 = np.expand_dims(logit / (1+logit) ,1)

        #y = np.concatenate((prob_0,prob_1), axis = 1)

        return prob_1
    
    return X, fn(X), fn, np.arange(0,4), 'squared Expoenetial'

def generate_additive_labels(n_samples=100, n_features=10):
        X = generate_X(n_samples, n_features)

        def fn(X):
            logit = (-10 * np.sin(0.2*X[:,0]) + abs(X[:,1]) + X[:,2]) + np.exp(-X[:,3])  

            prob_1 = np.expand_dims(1 / (1+logit) ,1)
            prob_0 = np.expand_dims(logit / (1+logit) ,1)

            y = np.concatenate((prob_0,prob_1), axis = 1)
            return logit

        return X, fn(X), fn, np.arange(0,4), "Nonlinear Additive"

def generate_XOR(n_samples=100, n_features=10):
    X = generate_X(n_samples, n_features)
    
    def fn(X):
        y = 0.5 * ( np.exp(X[:,0]*X[:,1]*X[:,2]) + np.exp(X[:,3]*X[:,4]))
        prob_1 = np.expand_dims(1 / (1+y) ,1)

        return y

    return X, fn(X), fn, np.arange(0,5), "XOR data set"

def generate_simple_interactions(n_samples=100, n_features=10):
    X = generate_X(n_samples, n_features)
    
    def fn(X):
        y = ((X[:,0]*X[:,1]*X[:,2])) + 0.001 * np.random.normal(0, 1, X[:,0].shape) #+ (X[:,3]*X[:,4]))

        return y

    return X, fn(X), fn, np.arange(0,5), "simple interactions"
