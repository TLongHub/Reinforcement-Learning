import torch as t

def initialise():
    global Q, N, K
    Q = t.zeros(1, K) # Initiate Q as 0 everywhere
    N = 0 # Initialise N, the time step

def actual_reward(q_star, action):
    reward = t.normal(t.select(q_star, 1, action), 1) # Takes q^*(a) as mean and 1 as variance
    return reward

def SA_update(action, reward):
    global Q, N
    """Updates the estimates Q according to experienced rewards using sample average."""
    mu = t.select(Q, 1, action) # Q(a)
    Q[0, action] = t.tensor([(mu * N + reward)/(N + 1)]) # Update the Q(a) value
    N =+ 1 # Update n

def DF_update(action, reward, alpha):
    global Q, N
    """Updates the estimates Q according to experienced rewards using discounting factor, alpha."""
    mu = t.select(Q, 1, action) # Q(a)
    Q[0, action] = t.tensor([mu + alpha * (reward - mu)]) # Update the Q(a) value
    N =+ 1 # Update n

def greedy(q_star, alpha, DF = False):
    global Q, N, K
    """Performs greedy learning algorithm for k-bandit problem."""
    tc = 0 # Number of exploration cycles (select 0 for no exploration)
    n = 500 # Number of additional time steps
    for _ in range(tc): # Start exploration cycle
        for action in range(K): # Itterate over all possible actions
            reward = actual_reward(q_star, action)
            if DF == False:
                SA_update(action, reward)
            else:
                DF_update(action, reward, alpha)
    
    for _ in range(n): # Start exploitation cycles
        action = t.argmax(Q) # Selecet the most favoured action
        reward = actual_reward(q_star, action)
        if DF == False:
            SA_update(action, reward)
        else:
            DF_update(action, reward, alpha)
    print("The last action selected (favoured):", action)

def epsilon_greedy(q_star, alpha, epsilon, DF = False): # Very similar to greedy approach...
    global Q, N, K
    """Performs epsilon-greedy learning algorithm for k-bandit problem."""
    n = 500 # Number of additional time steps
    
    for _ in range(n):
        if t.rand(1) <= epsilon: # With probability epsilon, choose action at random
            action = t.randint(10, (1, 1))
        else:
            action = t.argmax(Q) # Selecet the most favoured action
        reward = actual_reward(q_star, action)
        if DF == False:
            SA_update(action, reward)
        else:
            DF_update(action, reward, alpha)
    print("The last action selected (favoured):", action)

