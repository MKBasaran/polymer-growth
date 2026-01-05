import numpy as np

def hillClimbing(diff_func, arg_func):
    param_names = ["time_sim","number_of_molecules", "monomer_pool", "p_growth", "p_death", "p_dead_react", "l_exponent", "d_exponent", "l_naked", "kill_spawns_new"]
    epsilon = 0.00002
    dims = 10
    # initial_step_sizes = np.ones(dims) * 0.2
    current_point = np.array([1000, 100000,     31600000,   0.2,
    0.0000806, 0.5, 0.67, 0.67, 1, 1])   # the zero-magnitude vector is common
    initial_step_sizes = current_point/10
    some_acceleration = 5.0

    step_size = initial_step_sizes   # a vector of all 1's is common
    acceleration = some_acceleration  # a value such as 1.2 is common
    candidate = np.zeros(5)
    candidate[0] = -acceleration
    candidate[1] = -1 / acceleration
    candidate[2] = 0
    candidate[3] = 1 / acceleration
    candidate[4] = acceleration
    iter = 0
    while True:
        iter+=1
        print("##### iter", iter, "#####")
        before = diff_func(arg_func(current_point), plot=True)
        for x in range(len(current_point)):
            print("{}:{}; ".format(param_names[x], current_point[x]))
        print("solution cost:", before)
        acc_lowered = False
        for i in range(len(current_point)):
            print("##### param", param_names[i], "#####")
            best = -1
            best_score = 100000
            for j in range(5):        # try each of 5 candidate locations
                current_point[i] = current_point[i] + step_size[i] * candidate[j]
                temp = diff_func(arg_func(current_point), plot=True)
                print("{}: val: {}, cost: {}".format(j, current_point[i], temp))
                current_point[i] = current_point[i] - step_size[i] * candidate[j]

                if temp < best_score:
                    best_score = temp
                    best = j
            print("keeping option {}".format(best))
            if candidate[best] == 0:
                step_size[i] = step_size[i] / acceleration
                acc_lowered = True
            else:
                current_point[i] = current_point[i] + step_size[i] * candidate[best]
                step_size[i] = step_size[i] * candidate[best]  # accelerate

        if abs(diff_func(arg_func(current_point)) - before) < epsilon and not acc_lowered:
            return current_point
