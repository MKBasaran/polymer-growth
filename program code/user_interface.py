import tkinter as tk
import matplotlib
import numpy as np
import tkinter.ttk as ttk
import time
import imageio

from GA_base import GA_base
from GA_base_NN import GA_base_NN
from GA_paretto import GA_paretto
from co_evolution import co_evolution
from cross_island_breeding_GA import cross_island_breeding_GA
from data_generator import data_generator
from island_GA import island_GA
from show_population import show_population
from simulation import polymer
from distributionComparison import medianFoldNorm, minMaxNorm, translationInvariant, basicCostFunction, \
    locationCostFunction, min_maxV2
from evolutionaryAlgorithm import EvolutionaryAlgorithm
from bayesianOptimization import bayesianOptimisation
from fddc import fddc

# TODO: algorithm names
# CONSTANTS
from tester import tester
from txt_to_excel import txt_to_excel

EA = "evolutionary algorithm"
BO = "bayesian optimization"
GA = "genetic algorithm"
GA_NN = "genetic algorithm NN"
IGA = "island GA"
PGA = "paretto GA"
SP = "show population"
CGA = "cross GA"
CE = "Co-evolution"
DG = "Data generator"
T = "Tester"
FDDC = "fitness diversity driven coevolution"


class Parameter:
    def __init__(self, name, t, value, widget=None):
        if not isinstance(value, t):
            raise ValueError("{} is not of type {}".format(value, t))
        self.name = name
        self.type = t
        self.value = value
        if widget is not None:
            self.set_var(widget)

    def determine_var(self, t):
        tk_vars = {int: tk.IntVar, float: tk.DoubleVar, str: tk.StringVar, bool: tk.BooleanVar}
        return tk_vars[t]

    def get_name(self):
        return self.name

    def get_type(self):
        return self.type

    def get_value(self):
        return self.var.get()

    def set_var(self, widget):
        self.var = self.determine_var(self.type)(widget, self.value, self.name)

    def get_var(self):
        return self.var


validation = {}


def register_validation_functions(widget):
    validation[int] = widget.register(is_int)
    validation[float] = widget.register(is_float)
    validation[str] = widget.register(is_str)
    validation[bool] = widget.register(is_bool)


def is_int(s_old, s_new, name):
    if not _try_cast(s_new, int):
        return False
    return True


def is_str(s_old, s_new, name):
    if not _try_cast(s_new, str):
        return False
    return True


def is_float(s_old, s_new, name):
    if not _try_cast(s_new, float):
        return False
    return True


def is_bool(s_old, s_new, name):
    return s_new.lower() in ['true', 'false', '0' '1']


def _try_cast(val, t):
    try:
        t(val)
        return True
    except ValueError:
        return False


# TODO: cost fucntions
MFC = "MedianFoldChange"
MM = "MaxMinNorm"
TIMFC = "Translation invariant MFC"
MMV2 = "Min max V2"
BF = "Basic function"
LF = "Location function"


class CostFunctionChooser:
    def __init__(self, parent):
        self.parent = parent
        lf = tk.LabelFrame(parent, text="difference function")
        file_parameter = Parameter("filename", str, "Data/5k no BB.xlsx")
        file_parameter.set_var(parent)
        sigma_parameter = Parameter("weights", str, "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1", parent)
        f_parameter = Parameter("transfac", float, 1.0, parent)
        # TODO: add choices
        self.parameters = {
            LF: [file_parameter],
            BF: [file_parameter],
            MFC: [file_parameter, sigma_parameter],
            TIMFC: [file_parameter, sigma_parameter, f_parameter],
            MMV2: [file_parameter, sigma_parameter, f_parameter],
            MM: [file_parameter]
        }
        self.function_list = tk.Listbox(lf, height=len(self.parameters), exportselection=0)
        self.functions = [MMV2, TIMFC, LF, BF, MM, MFC]
        [self.function_list.insert(tk.END, x) for x in self.functions]
        self.function_list.bind('<<ListboxSelect>>', self.switch)
        self.function_list.select_set(0)
        self.function_list.pack(side=tk.BOTTOM)
        lf.pack()

        param_frame = tk.LabelFrame(self.parent, text="parameters")
        param_frame.pack()
        self.function_frames = self.build_frames(self.parameters, param_frame)
        self.current_frame = None
        self.switch(None)

        self.parameter_frame = tk.LabelFrame(parent, text="parameters")
        self.parameter_frame.pack()

    def switch(self, event):
        if self.current_frame is not None:
            self.current_frame.pack_forget()

        index = self.function_list.curselection()[0]
        func = self.functions[index]
        new_frame = self.function_frames[func]
        new_frame.pack()
        self.current_frame = new_frame

    def build_frames(self, functions, parent_widget):
        frames = {}
        for function in functions:
            parent = tk.Frame(parent_widget)
            for param in self.parameters[function]:
                control = None
                param_type = param.get_type()
                if param_type in [int, str, float]:
                    lab = tk.Label(parent, text=param.get_name())
                    lab.pack()
                    control = tk.Entry(parent,
                                       validate='all',
                                       validatecommand=(validation[param_type], '%s', '%P', '%W'),
                                       name=param.get_name(),
                                       textvar=param.get_var())
                elif param_type is bool:
                    control = tk.Checkbutton(parent, text=param.get_name(), name=param.get_name())
                control.pack()
            frames[function] = parent
        return frames

    def get_function(self):
        index = self.function_list.curselection()[0]
        func = self.functions[index]
        return func, self.parameters[func]


class App(tk.Frame):
    def __init__(self, master=None):
        super(App, self).__init__(master)
        self.pack()

        self.fig = Figure()
        self.result_frame = tk.LabelFrame(self, text="result")
        self.plot_window = FigureCanvasTkAgg(self.fig, self.result_frame)
        self.plot_window.draw()
        self.plot_window.get_tk_widget().pack()
        self.result_frame.grid(column=2, row=0, rowspan=2)

        self.function_frame = tk.LabelFrame(self, text="cost function")
        self.function_chooser = CostFunctionChooser(self.function_frame)
        self.function_frame.grid(column=1, row=1)

        # param_boundaries = np.array([[900, 1100], [90000, 110000], [3000000, 32000000],
        #                              [0, 1], [0, 0.0001], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]])
        # TODO: intitalize to actually usefull values
        # set variables
        self.lower_parameters = [
            Parameter("l_time_sim", int, 100),  # origanally 1000
            Parameter("l_number_of_molecules", int, 100),  # origanally 100000
            Parameter("l_monomer_pool", int, 10000),  # origanally 3100000
            Parameter("l_p_growth", float, 0.0),  # origanally 0.5
            Parameter("l_p_death", float, 0.00000),  # origanally 0.00005
            Parameter("l_p_dead_react", float, 0.0),  # origanally 0.5
            Parameter("l_l_exponent", float, 0.0),  # origanally 0.5
            Parameter("l_d_exponent", float, 0.0),  # origanally 0.5
            Parameter("l_l_naked", float, 0.0),  # origanally 0.5
            Parameter("l_kill_spawns_new", bool, False)  # origanally True
        ]

        self.upper_parameters = [
            Parameter("u_time_sim", int, 5000),  # origanally 1000
            Parameter("u_number_of_molecules", int, 100000),  # origanally 100000
            Parameter("u_monomer_pool", int, 5000000),  # origanally 3100000
            Parameter("u_p_growth", float, 1.0),  # origanally 0.5
            Parameter("u_p_death", float, 0.010),  # origanally 0.00005
            Parameter("u_p_dead_react", float, 1.0),  # origanally 0.5
            Parameter("u_l_exponent", float, 1.0),  # origanally 0.5
            Parameter("u_d_exponent", float, 1.0),  # origanally 0.5
            Parameter("u_l_naked", float, 1.0),  # origanally 0.5
            Parameter("u_kill_spawns_new", bool, True)  # origanally True
        ]

        # corrected values:
        number_of_molecules = 50000
        monomer_pool = number_of_molecules * 50
        self.corrected_lower_parameters = [
            Parameter("l_time_sim", int, 100),  # origanally 1000
            Parameter("l_number_of_molecules", int, 10000),  # origanally 100000
            Parameter("l_monomer_pool", int, 10000),  # origanally 3100000
            Parameter("l_p_growth", float, 0.2),  # origanally 0.5
            Parameter("l_p_death", float, 0.00000),  # origanally 0.00005
            Parameter("l_p_dead_react", float, 0.0),  # origanally 0.5
            Parameter("l_l_exponent", float, 0.5),  # origanally 0.5
            Parameter("l_d_exponent", float, 0.5),  # origanally 0.5
            Parameter("l_l_naked", float, 0.0),  # origanally 0.5
            Parameter("l_kill_spawns_new", bool, True)  # origanally True
        ]

        self.corrected_upper_parameters = [
            Parameter("u_time_sim", int, 100000),  # origanally 1000
            Parameter("u_number_of_molecules", int, 50000),  # origanally 100000
            Parameter("u_monomer_pool", int, 5000000),  # origanally 3100000
            Parameter("u_p_growth", float, 0.2),  # origanally 0.5
            Parameter("u_p_death", float, 0.00020),  # origanally 0.00005
            Parameter("u_p_dead_react", float, 1.0),  # origanally 0.5
            Parameter("u_l_exponent", float, 0.8),  # origanally 0.5
            Parameter("u_d_exponent", float, 0.8),  # origanally 0.5
            Parameter("u_l_naked", float, 1.0),  # origanally 0.5
            Parameter("u_kill_spawns_new", bool, True)  # origanally True
        ]

        # self.upper_parameters = [Parameter("u_" + x.get_name()[2:], x.get_type(), x.value) for x in self.lower_parameters]
        # EA: #iterations, population size, fitness_function
        pop = 50
        it = 30
        self.alg_parameters = {
            EA: [Parameter("#iterations", int, value=it),
                 Parameter("population_size", int, pop)
                 # ,Parameter("distribution file location", str, "Data\polymer_20k.xlsx")
                 ],
            BO: [Parameter("#iterations", int, it)],
            GA: [Parameter("#iterations", int, value=it),
                 Parameter("population_size", int, pop),
                 Parameter("load fom file", bool, False),
                 Parameter("load_from file_name", str, "fakeData/sim_val0")
                 ],
            GA_NN: [Parameter("#iterations", int, value=it),
                    Parameter("population_size", int, pop),
                    Parameter("load fom file", bool, False),
                    Parameter("load_from file_name", str, "fakeData/sim_val0")
                    ],
            PGA: [Parameter("#iterations", int, value=it),
                  Parameter("population_size", int, pop)
                  ],
            IGA: [Parameter("#iterations", int, value=it),
                  Parameter("population_size", int, pop),
                  Parameter("islands", int, 4),
                  Parameter("load fom file", bool, True),
                  Parameter("generated data file name", str, 'fakeData/sim_val0')
                  ],
            T: [Parameter("#iterations", int, value=it),
                Parameter("population_size", int, pop),
                Parameter("load fom file", bool, True),
                Parameter("generated data file name", str, 'fakeData/sim_val0'),
                Parameter("save as", str, 'name')

                ],
            FDDC: [Parameter("#iterations", int, value=it),
                   Parameter("population_size", int, pop),
                   Parameter("load fom file", bool, True),
                   Parameter("file_name", str, 'fakeData/sim_val0')
                   ],
            CGA: [Parameter("#iterations", int, value=it),
                  Parameter("population_size", int, pop)
                  ],
            SP: [Parameter("#iterations", int, value=it),
                 Parameter("population_size", int, pop)
                 ],
            CE: [Parameter("#iterations", int, value=it),
                 Parameter("population_size", int, pop),
                 Parameter("load fom file", bool, True),
                 Parameter("load_from file_name", str, "fakeData/sim_val0")
                 ],
            DG: [Parameter("population_size", int, pop)]
            # TODO: set parameters
        }
        [x.set_var(self) for x in self.lower_parameters]
        [x.set_var(self) for x in self.upper_parameters]

        self.build_alg_selector()
        self.build_widgets()

    def build_alg_selector(self):
        lf = tk.LabelFrame(self, text="algorithm")
        lf.grid(column=1, row=0, sticky=tk.N)
        self.alg_list = tk.Listbox(lf, height=2, exportselection=0)
        self.alg_list.insert(tk.END, T)
        self.alg_list.insert(tk.END, IGA)
        self.alg_list.insert(tk.END, CE)
        self.alg_list.insert(tk.END, FDDC)
        self.alg_list.insert(tk.END, PGA)
        self.alg_list.insert(tk.END, CGA)
        self.alg_list.insert(tk.END, GA_NN)
        self.alg_list.insert(tk.END, GA)
        self.alg_list.insert(tk.END, DG)
        self.alg_list.insert(tk.END, SP)
        self.alg_list.insert(tk.END, EA)
        self.alg_list.insert(tk.END, BO)
        # TODO: add new GA
        self.alg_list.bind('<<ListboxSelect>>', self._switch_alg_parameters)
        self.alg_list.pack()
        # Force the settings to be shown
        # build frames for each alg
        self.alg_frames = {}
        for alg in self.alg_parameters:
            f = tk.LabelFrame(lf, text=alg)
            params = self.alg_parameters[alg]
            for param in params:
                param.set_var(self)
                param_type = param.get_type()
                lab = tk.Label(f, text=param.get_name())
                lab.pack()
                control = tk.Entry(f,
                                   validate='all',
                                   validatecommand=(validation[param_type], '%s', '%P', '%W'),
                                   name=param.get_name(),
                                   textvar=param.get_var())
                control.pack()
            self.alg_frames[alg] = f
        self.current_alg_frame = None
        self.alg_list.select_set(0)
        self._switch_alg_parameters(None)

    def _switch_alg_parameters(self, event):
        # For some reason this method is called when tab is pressed in other fields.... :(
        try:
            selected = self.alg_list.get(self.alg_list.curselection()[0])
        except IndexError:
            return
        if self.current_alg_frame is not None:
            self.current_alg_frame.pack_forget()
        frame = self.alg_frames[selected]
        frame.pack()  # column=1, row=1)
        self.current_alg_frame = frame

    def build_widgets(self):
        self.param_controls = {}
        self.parameter_frame = tk.LabelFrame(self, text="initial parameters")
        self.lower_frame = tk.LabelFrame(self.parameter_frame, text="lower bounds")
        self.upper_frame = tk.LabelFrame(self.parameter_frame, text="upper bounds")
        self.parameter_frame.grid(column=0, row=0, rowspan=2)

        self.lower_frame.grid(column=0, row=0)
        self.upper_frame.grid(column=1, row=0)
        self.create_parameter_controls(self.lower_parameters, self.lower_frame)
        self.create_parameter_controls(self.upper_parameters, self.upper_frame)

        self.run = tk.Button(self, text="Run algorithm", command=self.run)
        self.run.grid(column=0, columnspan=2)

    def create_parameter_controls(self, params, parent):
        for param in params:
            control = None
            param_type = param.get_type()
            if param_type in [int, str, float]:
                lab = tk.Label(parent, text=param.get_name())
                lab.pack()
                control = tk.Entry(parent,
                                   validate='all',
                                   validatecommand=(validation[param_type], '%s', '%P', '%W'),
                                   name=param.get_name(),
                                   textvar=param.get_var())
            elif param_type is bool:
                control = tk.Checkbutton(parent, text=param.get_name(), name=param.get_name())

            self.param_controls[str(control)] = control
            control.pack()

    def run(self):
        selected_alg = self.alg_list.get(self.alg_list.curselection()[0])
        alg_parameters = [x.get_value() for x in self.alg_parameters[selected_alg]]

        low = np.array([x.get_value() for x in self.lower_parameters])
        high = np.array([x.get_value() for x in self.upper_parameters])

        [x.set_var(self) for x in self.corrected_lower_parameters]
        [x.set_var(self) for x in self.corrected_upper_parameters]

        corrected_low = np.array([x.get_value() for x in self.corrected_lower_parameters])
        corrected_high = np.array([x.get_value() for x in self.corrected_upper_parameters])
        print(selected_alg)
        func, func_alg = self.function_chooser.get_function()
        # TODO: fitness
        if func == MFC:
            file_name = func_alg[0].get_value()
            sigma_str = func_alg[1].get_value()
            sigma = np.array([float(x) for x in sigma_str.split(",")])
            norm = medianFoldNorm(file_name, polymer, sigma, fig=self.fig)
        elif func == MM:
            file_name = func_alg[0].get_value()
            norm = minMaxNorm(file_name, polymer, fig=self.fig)
        elif func == BF:
            file_name = func_alg[0].get_value()
            norm = basicCostFunction(file_name, polymer, fig=self.fig)
        elif func == LF:
            file_name = func_alg[0].get_value()
            norm = locationCostFunction(file_name, polymer, fig=self.fig)
        elif func == TIMFC:
            file_name = func_alg[0].get_value()
            sigma_str = func_alg[1].get_value()
            sigma = np.array([float(x) for x in sigma_str.split(",")])
            transfac = func_alg[2].get_value()
            norm = translationInvariant(file_name, polymer, sigma, transfac, fig=self.fig)
        elif func == MMV2:
            file_name = func_alg[0].get_value()
            sigma_str = func_alg[1].get_value()
            sigma = np.array([float(x) for x in sigma_str.split(",")])
            transfac = func_alg[2].get_value()
            norm = min_maxV2(file_name, polymer, sigma, transfac, fig=self.fig)

        bound = np.stack((low, high), 1)
        corrected_bound = np.stack((corrected_low, corrected_high), 1)
        # TODO: selected alg stuff
        if selected_alg == EA:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]

            ea = EvolutionaryAlgorithm(bound, pop_size, norm.costFunction)
            for i in range(iterations):
                print("iteration", i)
                f = ea.run(1)
                self.plot_window.draw()
                result = ea.get_best_individual(f)
                self.update()
        elif selected_alg == BO:
            iterations = alg_parameters[0]
            result = bayesianOptimisation(iterations, norm.costFunction, bound, 10)
        elif selected_alg == FDDC:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]
            read_from_file = alg_parameters[2]
            read_from_file_name = alg_parameters[3]

            fc = fddc(bounds=bound, fitnessFunction=norm.costFunction, distribution_comparison=norm,
                      populationSize=pop_size, read_from_file=read_from_file, read_from_file_name=read_from_file_name)
            for i in range(iterations):
                print("iteration", i)
                fc.run()
                self.plot_window.draw()
                result = fc.best
                self.update()
        elif selected_alg == GA:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]
            read_from_file = alg_parameters[2]
            read_from_file_name = alg_parameters[3]
            ga = GA_base(corrected_bound, norm.costFunction, distribution_comparison=norm, populationSize=pop_size,
                         read_from_file=read_from_file, read_from_file_name=read_from_file_name)
            for i in range(iterations):
                print("iteration", i)
                ga.run()
                self.plot_window.draw()
                result = ga.best
                self.update()
        elif selected_alg == GA_NN:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]
            read_from_file = alg_parameters[2]
            read_from_file_name = alg_parameters[3]
            ga_nn = GA_base_NN(bound, norm.costFunction, distribution_comparison=norm, populationSize=pop_size,
                               read_from_file=read_from_file, read_from_file_name=read_from_file_name)
            for i in range(iterations):
                print("iteration", i)
                ga_nn.run()
                self.plot_window.draw()
                result = ga_nn.best
                self.update()
        elif selected_alg == IGA:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]
            islands = alg_parameters[2]
            read_from_file = alg_parameters[3]
            read_from_file_name = alg_parameters[4]

            file_name = func_alg[0].get_value()
            transfac = func_alg[2].get_value()
            new_sigma = "2,2,1,1,2,2"
            sigma2 = np.array([float(x) for x in new_sigma.split(",")])
            norm2 = translationInvariant(file_name, polymer, sigma2, transfac, fig=self.fig)

            iga = island_GA(bound, corrected_bound, norm.costFunction, norm2.costFunction, norm,
                            populationSize=pop_size,
                            number_of_islands=islands, read_from_file=read_from_file,
                            read_from_file_name=read_from_file_name)
            for i in range(iterations):
                print("iteration", i)
                iga.run()
                self.plot_window.draw()
                result = iga.best
                self.update()

        # generally always use this as it saves results
        elif selected_alg == T:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]
            read_from_file = alg_parameters[2]
            read_from_file_name = alg_parameters[3]
            save_as = alg_parameters[4]

            # create tester class
            t = tester(bounds=bound, corrected_bounds=corrected_bound, fitnessFunction=norm.costFunction,
                       distribution_comparison=norm, populationSize=pop_size, number_of_generations=iterations,
                       read_from_file=read_from_file,read_from_file_name=read_from_file_name)

            # start the run
            name = save_as
            start = time.time()
            for p in range(1):
                # create file to save resutts
                file = open('results/{}.txt'.format(name), 'w')
                file.close()
                t.create_GA()
                current_best_score = 100000

                # the run loop
                for i in range(iterations):
                    t.run()
                    self.plot_window.draw()
                    result = t.best

                    if t.current_results[len(t.current_results) - 1] < current_best_score:
                        current_best_score = t.current_results[len(t.current_results) - 1]

                    # save result ever generation to allow the user to quit during run time
                    file = open('results/{}.txt'.format(name), 'a')
                    file.write('{}, '.format(str(current_best_score)))
                    file.close()
                    self.update()

                    # keep running untill a certain score is reached or one day has passed
                    if t.current_results[len(t.current_results) - 1] < 1 or (time.time() - start) > (172800/2):
                        break

                # store best individual and create excel file of all costs
                t.store_best()
                txt_to_excel('results/{}'.format(name))
                file = open('results/{}.txt'.format(name), 'a')
                file.write(' best: {}'.format(t.best))
                file.close()
            print('final time: {}'.format(time.time() - start))

        elif selected_alg == PGA:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]

            file_name = func_alg[0].get_value()
            transfac = func_alg[2].get_value()
            new_sigma = "2,2,1,1,2,2"
            sigma2 = np.array([float(x) for x in new_sigma.split(",")])
            norm2 = translationInvariant(file_name, polymer, sigma2, transfac, fig=self.fig)

            pga = GA_paretto(bound, corrected_bound, norm.costFunction, norm2.costFunction, populationSize=pop_size)
            for i in range(iterations):
                print("iteration", i)
                pga.run()
                self.plot_window.draw()
                result = pga.best
                self.update()
        elif selected_alg == CGA:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]

            cga = cross_island_breeding_GA(bound, corrected_bound, norm.costFunction, populationSize=pop_size)
            for i in range(iterations):
                print("iteration", i)
                cga.run()
                self.plot_window.draw()
                result = cga.best
                self.update()
        elif selected_alg == SP:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]
            sp = show_population(bound, norm.costFunction, populationSize=pop_size)
            for i in range(iterations):
                print("iteration", i)
                sp.run()
                self.plot_window.draw()
                result = sp.best
                self.update()

        elif selected_alg == CE:
            iterations = alg_parameters[0]
            pop_size = alg_parameters[1]
            read_from_file = alg_parameters[2]
            read_from_file_name = alg_parameters[3]
            ce = co_evolution(bound, corrected_bound, norm.costFunction, norm, populationSize=pop_size,
                              read_from_file=read_from_file, read_from_file_name=read_from_file_name)
            for i in range(iterations):
                print("iteration", i)
                ce.run()
                self.plot_window.draw()
                result = ce.best
                self.update()
        if selected_alg == DG:
            pop_size = alg_parameters[0]
            dg = data_generator(bound, norm.costFunction, populationSize=pop_size)
            dg.run()
            self.plot_window.draw()
            result = dg.best
            self.update()
        # TODO:add new alg
        # print([x.get_value() for x in self.alg_parameters[selected_alg]])
        x = tk.Text(self, height=2)
        x.delete(0.1, tk.END)
        x.insert(tk.END, str(result))

        x.grid(column=0, columnspan=2, row=3)


class SimulationPage(tk.Frame):

    def __init__(self, master=None):
        super(SimulationPage, self).__init__(master)
        self.parameter_frame = tk.LabelFrame(self, text="parameters")
        self.build_parameters()

        self.result_frame = tk.LabelFrame(self, text="result")
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.plot_window = FigureCanvasTkAgg(self.fig, self.result_frame)
        self.plot_window.draw()
        self.plot_window.get_tk_widget().pack()
        self.pack()

        self.run = tk.Button(self, text="Run algorithm", command=self.run)
        self.parameter_frame.grid(column=0, row=0)
        self.run.grid(column=0, row=1)
        self.result_frame.grid(column=1, row=0, columnspan=2, rowspan=2)

    def build_parameters(self):
        self.video = Parameter("video", bool, True)
        self.save_video = Parameter("save video", bool, True)
        self.parameters = [
            Parameter("time_sim", int, 1000),
            Parameter("number_of_molecules", int, 100000),
            Parameter("monomer_pool", int, 3100000),
            Parameter("p_growth", float, 0.5),
            Parameter("p_death", float, 0.00005),
            Parameter("p_dead_react", float, 0.5),
            Parameter("l_exponent", float, 0.5),
            Parameter("d_exponent", float, 0.5),
            Parameter("l_naked", float, 0.5),
            Parameter("kill_spawns_new", bool, True),
            self.video,
            self.save_video
        ]

        [x.set_var(self) for x in self.parameters]
        for param in self.parameters:
            control = None
            param_type = param.get_type()
            if param_type in [int, str, float]:
                lab = tk.Label(self.parameter_frame, text=param.get_name())
                lab.pack()
                control = tk.Entry(self.parameter_frame,
                                   validate='all',
                                   validatecommand=(validation[param_type], '%s', '%P', '%W'),
                                   name=param.get_name(),
                                   textvar=param.get_var())
            elif param_type is bool:
                control = tk.Checkbutton(self.parameter_frame, text=param.get_name(), name=param.get_name())

            control.pack()

    def run(self):
        values = [x.get_value() for x in self.parameters if x.get_name() not in ["video", "save video"]]

        self.run["text"] = "running"
        self.update()
        if self.save_video.get_value() and self.video.get_value():
            name = str(round(time.time()))
            self.writer = imageio.get_writer(name + '.gif', mode='I', duration=0.3)
        if self.video.get_value():
            result = polymer(*values, UI_vid=self.make_hist)
        else:
            result = polymer(*values)
            self.make_hist(result)

        if self.save_video.get_value() and self.video.get_value():
            self.writer.close()

        self.run["text"] = "run"

    def make_hist(self, results, state=None, coloured=1):
        self.ax.clear()
        living, dead, coupled = results
        if state is not None:
            current_monomer, initial_monomer, time = state
            conversion = 1 - current_monomer / initial_monomer
        d = np.hstack((living, dead, coupled))
        DPn = np.mean(d)
        DPw = np.sum(np.square(d)) / (DPn * d.shape[0])
        PDI = DPw / DPn
        # dlmwrite('polymerOutput.txt',[time, conversion, DPn, DPw, PDI], '-append');
        if coloured == 0:
            self.ax.hist(d, bins=int(np.max(d) - np.min(d)), facecolor='b')
        else:
            step = np.ceil((np.max(d) - np.min(d)) / 1000)
            binEdges = np.arange(np.min(d) - 0.5, np.max(d) + 0.5, step)
            midbins = binEdges[0:-1] + (binEdges[1:] - binEdges[0:-1]) / 2
            if coupled.size == 0:
                c, b, e = self.ax.hist([dead, living], bins=midbins, histtype='barstacked', stacked=False,
                                       label=['Dead', 'Living'])
                # e[0]["color"] = "blue"
                # e[1]["color"] = "orange"
                matplotlib.pyplot.setp(e[0], color="blue")
                matplotlib.pyplot.setp(e[1], color="orange")
                # setp(e[0], color='blue')
                # setp(e[1], color='orange')


            else:
                self.ax.hist([coupled, dead, living], bins=midbins, histtype='bar', stacked=True,
                             label=['Terminated', 'Dead', 'Living'])

        self.ax.set_xlabel('Length in units')
        self.ax.set_ylabel('Frequency')
        digits = 3
        if state is not None:
            title = "conversion={}, t={}, DPI={}, DPn={}, DPw={}".format(
                round(conversion, digits), time, round(PDI, digits), round(DPn, digits), round(DPw, digits)
            )
            self.ax.set_title(title)

        self.ax.legend()
        self.plot_window.draw()

        if self.save_video.get_value():
            width, height = self.plot_window.get_width_height()
            image = np.fromstring(self.plot_window.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
            self.writer.append_data(image)


class ApplicationNotebook(ttk.Notebook):

    def __init__(self, master=None):
        super(ApplicationNotebook, self).__init__(master)
        register_validation_functions(self)
        self.add(App(self), text="optimization")
        self.add(SimulationPage(self), text="Simulation")
        self.pack()


if __name__ == '__main__':
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    # app = App(root)
    root.title("Optimization")
    app = ApplicationNotebook(root)
    root.mainloop()
