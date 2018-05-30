import numpy as np
from joblib import Parallel, delayed
import multiprocessing

MAX_RESOURCE = 10
NUM_RESOURCES = 4

alpha = 0.1
max_occ = np.array([5,5,8,2])

player_structs = np.ones((4,4)) + np.eye(4)*3

def policyImprovement(i, occ, J):
    mu = np.arange(4, dtype=int)
    order = MAX_RESOURCE ** np.arange(NUM_RESOURCES)
    # incr is 1 if max_occ is larger than occ (i.e. max_occ hasn't been met)
    # and is 0 otherwise
    print('policyEvaluation:', mu.size)
    print('policyEvaluation:', occ[:,mu].size)
    incr = np.argmin([max_occ[mu], occ[mu]], axis=0) * player_structs[i % NUM_RESOURCES]
    print('project.py:policyImprovement(): player ', i)
    print('project.py:policyImprovement(): increment ', incr)
    print('project.py:policyImprovement(): occupancy ', occ)

    x = np.arange(np.size(J))
    x = np.transpose(np.tile(x, (NUM_RESOURCES,1)))

    x_next = x - (x / order % MAX_RESOURCE).astype(int) * order
    x_next = x_next + np.max([np.zeros((np.size(J), NUM_RESOURCES)), 
                              (x/order%MAX_RESOURCE-incr).astype(int)*order], axis=0)
    x_next = x_next.astype(int)

    print('project.py:policyImprovement():', x_next[9999,:])

    H = np.transpose(np.sum((np.transpose(np.tile(x_next, (NUM_RESOURCES,1,1)), (2,1,0))/order%MAX_RESOURCE).astype(int), axis=2))
    # print(H)
    print('project.py:policyImprovement():', H[9999,:])
    H = np.transpose(H ** 2 + alpha*J[x_next])
    print('project.py:policyImprovement():', H[:,9999])
    # print('project.py:policyImprovement:', H[:,9999])
    return (np.min(H, axis=0), np.argmin(H, axis=0))

def policyEvaluation(i, occ, mu, J, V):
    # H(x,mu,J) = g(x,mu) + a*J(f(x,mu))
    # J[t+1, i] = min(V, H(x,mu,J[t, :]))
    order = MAX_RESOURCE ** mu
    print('policyEvaluation:', mu.size)
    print('policyEvaluation:', occ[mu.astype(int)].size)
    incr = np.argmin([max_occ[mu.astype(int)], occ[mu.astype(int)]], axis=0) * player_structs[i % NUM_RESOURCES, mu.astype(int)]
    # print (i,occ, incr)
    x = np.arange(np.size(J))
    x_next = x - (x / order % MAX_RESOURCE).astype(int) * order
    x_next = x_next + np.max([np.zeros(np.size(J)), 
                              (x/order%MAX_RESOURCE-incr).astype(int)*order], axis=0)
    x_next = x_next.astype(int)
    # print('project.py:policyEvaluation:', x_next.shape)
    # print('project.py:policyEvaluation:', np.sum((np.tile(x_next, (4,1))/order%MAX_RESOURCE).astype(int), axis=0).shape)
    H = np.sum((np.tile(x_next, (NUM_RESOURCES,1))/order%MAX_RESOURCE).astype(int), axis=0) ** 2 + alpha*J[x_next]
    # print('project.py:policyEvaluation:', H[9999])
    return np.min([V, H], axis=0)

def main():
    time_horizon = 100
    num_players = 12
    num_resources = NUM_RESOURCES


    T = np.random.randint(2, size=(time_horizon, num_players))
    T_bar = np.random.randint(2, size=(time_horizon, num_players))
    
    # Policy will be per player, dependent on x = R_j - S_j, and is
    # initialized to be uniformly-distributed across all resources
    policy = np.zeros((time_horizon, num_players, MAX_RESOURCE ** num_resources))
    J = np.zeros((time_horizon, num_players, MAX_RESOURCE ** num_resources))
    V = (time_horizon*NUM_RESOURCES*MAX_RESOURCE)**2*np.ones((num_players, MAX_RESOURCE ** num_resources))
    
    occ = np.zeros((MAX_RESOURCE ** num_resources, NUM_RESOURCES))

    for t in range(time_horizon-1):
        for i in range(num_players):
            if T_bar[t,i] > 0:
                (J[t+1,i], policy[t+1,i]) = policyImprovement(i, occ, J[t,i])
                V[i] = J[t+1,i]
            elif T[t,i] > 0:
                J[t+1,i] = policyEvaluation(i, occ, policy[t,i], J[t,i], V[i])
                policy[t+1,i] = policy[t,i]
            else:
                J[t+1,i] = J[t,i]
                policy[t+1,i] = policy[t,i]
            # occ[policy.astype(int)[t+1,i]] = occ[policy.astype(int)[t+1,i]] + 1
            temp = np.tile(np.arange(NUM_RESOURCES), (MAX_RESOURCE ** NUM_RESOURCES, 1))
            occ = occ + np.transpose(np.transpose(temp) == policy)
            print('project.py:main(): policy ', policy.astype(int)[t+1,i])
            print('project.py:main(): occ ', occ)
        occ = np.zeros(NUM_RESOURCES)
        break
    np.save('J.npy', J)
    np.save('mu.npy', policy)

main()
