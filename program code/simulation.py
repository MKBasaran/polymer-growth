import numpy as np
import matplotlib.pyplot as plt
import time


# Set seed for 'Mersenne Twister' rng
# np.random.seed(2)
def polymer(time_sim, number_of_molecules, monomer_pool, p_growth, p_death, p_dead_react, l_exponent, d_exponent,
            l_naked, kill_spawns_new, video=0, coloured=1, final_plot=0, UI_vid=None):
    # this function simulates the growth of polymers, it takes;
    # number_of_molecules - the number of starting chains (length 1)
    # time_sim - the number of timesteps the simulation runs for
    # p_growth - the probability of growth for each monomer, each time step
    # p_death - the probability of a monomer dying and joining the dead pool
    # p_dead_react - the probilty a dead polymer reacts with a living one
    # killer_pool_size - the size of the pool of killer molecules in the system
    # p_killer - the probability a killer molecule reacts with a polymer
    # killer_pool_reusable - 0 means the killer pool is depleted every time a
    # killer molecule is used, 1 means the killer pool is never used up
    # kill_spawns_new - a binary flag as to whether a kill event means a new
    # polymer of length 1 is spawned or not 1=killing event spawns a new
    # polymer chain, 0 means it doesn't.
    # video - binary flag on whether to output to video
    # coloured - binary flag on whether to output the living, dead and coupled
    # polymers as separate colours in a stacked histogram.
    # monomer_pool - the size of the monomer pool, if it is a negative number
    # then no monomer pool is used and monomer supply is assumed infinite,
    # p_growth is now p_growth as provided * monomer_pool size
    # l_exponent - the exponent of the living length in the probability
    # calculation for successful coupling
    # d_exponent - the exponent of the dead length in the probability
    # calculation for successful coupling
    #
    #
    # It returns an array with the lengths of all polymers in the system
    # (living and dead)

    # Now let's write a subfunction to make the histogram
    # Prepare interactive plot
    # plt.ion()
    if final_plot:
        ax = plt.subplot(111)

    def make_histogram(living, dead, coupled, coloured, initial_monomer, current_monomer, time):

        ax.clear()
        d = np.hstack((living, dead, coupled))
        # calculate M_n, M_w and PDI
        DPn = np.mean(d)
        DPw = np.sum(np.square(d)) / (DPn * d.shape[0])
        PDI = DPw / DPn
        conversion = 1 - current_monomer / initial_monomer
        # dlmwrite('polymerOutput.txt',[time, conversion, DPn, DPw, PDI], '-append');
        if coloured == 0:
            ax.hist(d, bins=int(np.max(d) - np.min(d)), facecolor='b')
        else:
            step = np.ceil((np.max(d) - np.min(d)) / 1000)
            binEdges = np.arange(np.min(d) - 0.5, np.max(d) + 0.5, step)
            midbins = binEdges[0:-1] + (binEdges[1:] - binEdges[0:-1]) / 2
            if coupled.size == 0:

                c, b, e = ax.hist([dead, living], bins=midbins, histtype='barstacked', stacked=False,
                                  label=['Dead', 'Living'])
                plt.setp(e[0], color='blue')
                plt.setp(e[1], color='orange')


            else:
                ax.hist([coupled, dead, living], bins=midbins, histtype='bar', stacked=True,
                        label=['Terminated', 'Dead', 'Living'])

        ax.set_xlabel('Length in units')
        ax.set_ylabel('Frequency')
        ax.set_title(['conversion=', conversion, 'time=', time, 'PDI=', PDI, 'DPn=', DPn, 'DPw=', DPw])
        ax.legend()

        plt.pause(1e-40)

    # setup file for output

    # file = open('polymerOutput.txt', 'w')
    # print(file, '%s\n', 'Time, Conversion, DPn, DPw, PDI')
    # file.close()
    # the pool of living polymers is an array of 1's of the right size
    living = np.ones(int(number_of_molecules))

    # the pool of dead polymers
    dead = np.array([])
    coupled = np.array([])
    new_dead = np.array([])
    # if we're outputing video, set it all up
    if video == 1:
        pass
    # v=VideoWriter('polymer_distribution_video.avi');
    # open(v);
    initial_monomer_pool = monomer_pool
    for t in range(int(time_sim)):
        if living.shape[0] > 0:
            # first make a random vector with uniform random numbers which will
            # decide the fate of each living polymer
            r = np.random.random(living.shape[0])  # .astype('float32')

            # Now if the random number for a polymer is below p_growth, it will grow.
            ##### LIVING GROWTH #####
            if monomer_pool < 0:

                living[(r < p_growth)] = living[(r < p_growth)] + 1
                monomer_ratio = 1

            else:

                monomer_ratio = monomer_pool / initial_monomer_pool
                living[(r < (p_growth * monomer_ratio))] = living[(r < (p_growth * monomer_ratio))] + 1
                used_monomer = np.sum(r < (p_growth * monomer_ratio))

                if used_monomer > monomer_pool:
                    monomer_pool = 0
                else:
                    monomer_pool = monomer_pool - used_monomer
                # Next if the random number for a polymer is above p_growth but below
                # p_growth+(p_death*monomer_ratio) it will die store this in new dead for now, until
                # we've worked out if any of the old dead react, We multiply by
                # monomer ratio, as we believe monomer is also involved here.
                # However, if we have infinite monomer, monomer ratio is set to 1
                # above, so then this will be just p_death)
                ###### KILLING ######
            if kill_spawns_new == 1:

                # since the kill starts a new chain it uses up 1 monomer, so we
                # decrease the monomer pool
                if monomer_pool > 0:
                    new_dead = living[
                        (r < (p_growth * monomer_ratio + (p_death * monomer_ratio))) & (r >= p_growth * monomer_ratio)]
                    living[(r < (p_growth * monomer_ratio + (p_death * monomer_ratio))) & (
                            r >= p_growth * monomer_ratio)] = 1

                    number_killed = living[(r < (p_growth * monomer_ratio + (p_death * monomer_ratio))) & (
                            r >= p_growth * monomer_ratio)].shape[0]
                    if number_killed > monomer_pool:
                        monomer_pool = 0
                    else:
                        monomer_pool = monomer_pool - number_killed


            else:
                new_dead = living[
                    (r < (p_growth * monomer_ratio + (p_death * monomer_ratio))) & (r >= p_growth * monomer_ratio)]

                mask = np.ones(len(living), dtype=bool)
                mask[(r < (p_growth * monomer_ratio + (p_death * monomer_ratio))) & (
                            r >= p_growth * monomer_ratio)] = False
                living = living[mask]

                # living = np.delete(living, np.where((r < (p_growth * monomer_ratio + (p_death * monomer_ratio))) & (r >= p_growth * monomer_ratio)))
            # So in the new system each dead chain chooses another chain from
            # the system (either living or dead) to attack

            which_chain_attacked_per_dead = np.ceil(
                np.random.random(dead.shape[0]) * (dead.shape[0] + living.shape[0]) - 1).astype(int)

            # if the chosen chain is a dead one, we do nothing, so now only
            # consider when the chosen number is above the number of dead chains
            # let's implement this in a loop to consider each dead chain
            # individually

            still_dead = np.ones(dead.shape[0])
            # Dead that attack living chains as vampire, if it chooses to attack a living chain...
            dead_vampiric = (np.where(which_chain_attacked_per_dead > (dead.shape[0] - 1))[0])
            which_living_attacked = which_chain_attacked_per_dead[dead_vampiric] - dead.shape[0]

            # calculate the probability of success given the formula from Bryn
            r_success = np.random.random(which_living_attacked.shape[0])
            # # print(r_success.shape[0])
            # p_success = p_dead_react / (living[which_living_attacked] ** np.minimum(
            #             living[which_living_attacked] * (l_exponent / l_naked), l_exponent) * dead[dead_vampiric] ** np.minimum(
            #             dead[dead_vampiric] * (d_exponent / l_naked), d_exponent))  # the dead pool                                                                

            # groups formula
            p_success = p_dead_react / (np.float_power(living[which_living_attacked], np.minimum(
                living[which_living_attacked] * (l_exponent / l_naked), l_exponent)) * np.float_power(
                dead[dead_vampiric], np.minimum(
                    dead[dead_vampiric] * (d_exponent / l_naked), d_exponent)))  # the dead pool

            # Rachel's formula
            # p_success = p_dead_react / (living(which_living_attacked) ^ max(
            #     1 - living(which_living_attacked) * (1 - l_exponent) / l_naked, l_exponent) * dead(dead_counter) ^ (
            #                             1 - dead(dead_counter) * (1 - l_exponent) / l_naked, l_exponent))

            living[which_living_attacked[np.where(r_success < p_success)]] = (
                        living[which_living_attacked[np.where(r_success < p_success)]]
                        + dead[dead_vampiric[np.where(r_success < p_success)]])

            still_dead[dead_vampiric[np.where(r_success < p_success)]] = 0

            dead = dead[still_dead == 1]
            dead = np.hstack((dead, new_dead))

        if video == 1:
            make_histogram(living, dead, coupled, coloured, initial_monomer_pool, monomer_pool, t)

        if UI_vid is not None:
            UI_vid((living, dead, coupled), (monomer_pool, initial_monomer_pool, t))
        # frame=getframe(gcf);
        # writeVideo(v,frame);
    distribution = [living, dead, coupled]

    if final_plot:
        make_histogram(living, dead, coupled, coloured, initial_monomer_pool, monomer_pool, t)
        plt.show(block=True)

    # if video==1:
    # close(v)
    return distribution


if __name__ == '__main__':
    arg = [1.0e+03, 1.0e+05, 3.2e+07, 2.56761375e-01, 8.06000000e-05, 4.94705224e-03, 8.72555086e-01,
           4.06013255e-01, 3.84144228e-01, 1.00000000e+00]
    polymer(*arg, video=0, final_plot=1)

# Parameter Bryn
# polymer(1000, 100000, 31600000, 0.2,
# 0.0000806, 0.5, 0.67, 0.67, 1, 1, final_plot=1)

# np.save('output',polymer(1000, 100000, 100000000, .8, 0.0005, 0.2, 0.23, 0.23, 0.5, 1,final_plot=1))
# np.save('output',polymer(21000, 40000, 2000000, 0.99999, 0.25, 0.6000000000000001, 0.14000000000000007, 0.22800000000000006, 0.09936000000000009, 1,final_plot=1))
