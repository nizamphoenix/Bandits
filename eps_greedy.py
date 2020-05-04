from mab import MAB

class EpsGreedy(MAB):
    """
    Epsilon-Greedy multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    epsilon : float
        explore probability

    Q0 : float, optional
        initial value for the arms
    """
    def __init__(self, narms, epsilon, Q0=np.inf):
        self.narms = narms
        self.epsilon = epsilon
        self.Q0 = Q0
        self.counts_of_arms = np.zeros(narms)#k=narms+1
        self.Qvalues = np.full((narms,),Q0)#use infinity
    

    #This should return an arm integer in {1, . . . , self.narms}
    def play(self, tround, context=None):
        if np.random.random() > self.epsilon:#flipping a coin and chosing an arm at random
            return np.random.choice([i+1 for i, j in enumerate(self.Qvalues) if j == max(self.Qvalues)])
                   #Return the index(arm number) of the arm with the largest Q-value;in case of ties(more than 1 max value) pick a random index
        else:
            return 1+np.random.choice(range(len(self.Qvalues)))#chosing an arm based on calulated Qvalues

    """
    1.Increment the counts field that records the number of times we’ve played each of the arms.
    2.Find the current estimated value of the chosen arm.
      a)If this is our first experience with the chosen arm, we set the estimated value directly.
      b)If we had played the arm in the past, we update the estimated value of the chosen arm to be a weighted average of the
        previously estimated value and the reward we just received
    """
    def update(self, arm, reward, context=None):
        self.counts_of_arms[arm-1] += 1
        n = self.counts_of_arms[arm-1]
        if self.Qvalues[arm-1]==np.inf:#First experience with the arm.
            value=0                    #Done to avoid the issue with Nans'
        else:
            value = self.Qvalues[arm-1]#fetching the previous value of the arm 
        new_value = value * ((n - 1) / float(n)) + reward / float(n)
        self.Qvalues[arm-1] = new_value

#Epsilon-greedy is gullible when it is presented with arms with good rewards.
