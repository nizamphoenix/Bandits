from mab import MAB

class LinUCB(MAB):
    """
    Contextual multi-armed bandit (LinUCB) implenting algorithm in http://rob.schapire.net/papers/www10.pdf
    
    Arguments
    ========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context i.e length of context vector for each arm  

    alpha : float
        positive real explore-exploit parameter i.e. exploitation rate
    """
    
    def __init__(self, narms, ndims, alpha):
        self.narms = narms
        self.ndims = ndims
        self.alpha = float(alpha)
        self.theta = np.zeros((self.ndims,self.ndims,1))#An array of vectors to maintain coefficient estimates
        self.A = np.array([np.identity(self.ndims) for col in range(1, narms + 1)])#An array of length 10 of (10,10)identity matrices for each arm
        self.b = np.zeros((self.ndims,self.ndims,1))#An array of length 10 of (10,1)vectors
        
    def play(self, tround, context):
        posterior = [0 for col in range(1, self.narms + 1)]
        context_matrix = context.reshape((self.ndims, self.ndims))
        for arm in range(1, self.narms + 1):
            self.theta[arm - 1] = np.dot(inv(self.A[arm - 1]), self.b[arm - 1]) # Updating coefficient vector for an arm
            X = context_matrix[arm - 1].reshape((self.ndims, 1)) # Calculating X for each arm which is (10x1) vector
            stdev = np.math.sqrt(np.dot(np.dot(X.T , inv(self.A[arm - 1])) , X))  #standard deviation
            posterior[arm - 1] = (np.dot(self.theta[arm - 1].T ,X)) + self.alpha * stdev  #updating posterior(our belief about an arm) which was initialized to zero
        return np.random.choice([i for i, j in enumerate(posterior) if j == max(posterior)]) + 1  #chosing an arm at random and breaking the ties if they occur

    def update(self, arm, reward, context):
        context_matrix = context.reshape((self.ndims, self.ndims))#Reshaping the context of an event to ten(10,1) contexts, one for each arm.
        X=context_matrix[arm-1].reshape(self.ndims,1)#reshaping the context from (10,) to (10,1)
        self.A[arm-1] += np.dot(X,X.T)
        self.b[arm-1]+=reward * X
      
        
