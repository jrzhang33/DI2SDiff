import numpy as np
def cond_set(data_type, remain_rate, target,maxcond):
    if data_type == 'pamap':
        if remain_rate == 0.2:
            if maxcond == 2 and (target == 2 or target == 3):
                return [0.99,0.01]
            elif maxcond == 5 and (target == 1 or target == 0 or target ==3) :
                
                return [0.5, 1e-5, 1e-5, 1e-5, 0.5]
            elif maxcond == 10:
                return [0.5, 1e-05, 1e-05, 1e-05, 0.5, 0.5, 1e-05, 1e-05, 1e-05, 0.5]
            elif maxcond == 3:
                return [0.99, 0.01, 0.01]
            else:
                return np.ones(maxcond, dtype=int)
    elif data_type == 'dsads':
        if remain_rate == 0.2:
            if maxcond == 2 and target == 2:
                return [0.95,0.05]
            elif maxcond == 2 and target == 3:
                return [0.05,0.95]
            elif maxcond == 2:
                return [0.99,0.01]
            elif maxcond == 5 :
                return [0.5, 1e-5, 1e-5, 1e-5, 0.5]
            elif maxcond == 10:
                return [0.5, 1e-05, 1e-05, 1e-05, 0.5, 0.5, 1e-05, 1e-05, 1e-05, 0.5]
            elif maxcond == 3:
                return [0.99, 0.01, 0.01]
            else:
                return np.ones(maxcond, dtype=int)
    elif data_type == 'uschad':
        if remain_rate == 0.2:
            if maxcond == 2 and target == 2:
                return [0.95,0.05]
            elif maxcond == 3 and target == 1:
                return [1e-5, 0.95,0.05]
            elif maxcond == 3 and target == 2:
                return [0.1,0.9,0.01]
            elif maxcond == 3 and target == 3:
                return [0.5,0.5,0.1]
            # elif target == 3:
            #      return np.ones(maxcond, dtype=int)
            elif maxcond == 2:
                return [0.99,0.01]
            elif maxcond == 5 :
                return [0.5, 1e-5, 1e-5, 1e-5, 0.5]
            elif maxcond == 10:
                return [0.5, 1e-05, 1e-05, 1e-05, 0.5, 0.5, 1e-05, 1e-05, 1e-05, 0.5]
            elif maxcond == 3:
                return [0.99, 0.01, 0.01]
            elif maxcond == 15:
                return [1e-05, 1e-05, 1e-05, 0.5, 1e-5, 0.5, 1e-05, 1e-05, 1e-05, 0.5, 0.5, 1e-05, 1e-05, 1e-05, 0.5]
    else:
        return np.ones(maxcond, dtype=int)

