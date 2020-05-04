from mab import MAB

class UCB(MAB):
    """
    Upper Confidence Bound (UCB) multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    rho : float
        positive real explore-exploit parameter

    Q0 : float, optional
        initial value for the arms
    """
    

    def __init__(self, narms, rho, Q0=np.inf):
        self.narms = narms
        self.counts_of_arms = [0 for i in range(1,self.narms+1)]
        self.Qvalues = [Q0 for i in range(1,self.narms+1)]#use infinity
        self.Q0 = Q0
        self.rho=rho
        
    def play(self, tround, context=None):#No arm is played randomly
        for arm in range(1,self.narms+1):
            if self.counts_of_arms[arm-1] == 0:
                return arm
        ucbvalues = [0.0 for arm in range(1,self.narms+1)]
        total_counts = sum(self.counts_of_arms)
        for arm in range(1,self.narms+1):
            optimum_factor = np.math.sqrt((self.rho * np.math.log(total_counts)) / float(self.counts_of_arms[arm-1]))
            ucbvalues[arm-1] = self.Qvalues[arm-1] + optimum_factor
        return 1+np.random.choice(np.argwhere(self.Qvalues == np.amax(self.Qvalues)).flatten().tolist())
        #Return the index(arm number) of the arm with the largest Q-value;in case of ties(more than 1 max value) pick a random index
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
        if self.Qvalues[arm-1]==np.inf:#This is done to avoid Nan issue.
            value=0
        else:
            value = self.Qvalues[arm-1]#fetching the previous value of the arm 
        new_value = value * ((n - 1) / float(n)) + reward / float(n)# Recompute the estimated value of chosen arm using new reward
        self.Qvalues[arm-1] = new_value
    """
    UCB is not gullible unlike Epsilon greedy, rather it is optimistic for those arms for which the count is less.
    UCB thinks that the arms with less counts have not been 'explored' enough;this scenario occurs because 
    math.log(total_counts))/ float(self.counts[arm]) becomes large for arms that we know little about. 
    That means we try hard to learn about arms if we don’t know enough about them, even if they seem a little worse than the best arm.
    """

