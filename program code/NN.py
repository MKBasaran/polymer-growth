import numpy as np
import random as rand


# a basic fully connected NN
class NN:
    def __init__(self, input_size, output_size, number_of_hidden_layers, hidden_layer_size):
        self.input_size = input_size
        self.output_size = output_size
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_layer_size = hidden_layer_size

        # column number equals neuron number in next layer
        self.weights = list()
        for i in range(self.number_of_hidden_layers + 1):
            l1 = list()
            current_layer_size = self.hidden_layer_size
            if i is 0:
                current_layer_size = input_size

            for j in range(current_layer_size):
                l2 = list()
                next_layer_size = self.hidden_layer_size
                if i is self.number_of_hidden_layers:
                    next_layer_size = self.output_size

                for k in range(next_layer_size):
                    l2.append(rand.uniform(-1, 1))
                l1.append(l2)
            self.weights.append(l1)

    def compute_output(self, inp):
        output = list()
        current_output = inp

        for i in range(len(self.weights)):
            w = 0
            for k in range(len(self.weights[i][0])):
                s = 0
                for j in range(len(self.weights[i])):
                    # print('lw: {}, lco: {}, i: {}, j: {}, w: {} '.format(len(self.weights[i][j]), len(current_output),i,j,w))
                    s += self.weights[i][j][w] * current_output[j]
                # print('s {}'.format(s))
                s = 1 / (1 + np.exp(-s))
                output.append(s)
                w += 1

            current_output = output.copy()
            output = list()
        return current_output

if __name__ == '__main__':
    # NN = NN(113, 10, 2, 61)
    # print(NN.weights)
    # print(NN.compute_output(
    #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0014584681084496814, 0.0017425013424798047, 0.0021232316974865474, 0.0025647665828151087,
    #      0.003064102260131324, 0.0036134396222747417, 0.004200008603830057, 0.0048065859263884195, 0.005412766433497137,
    #      0.005996917436980615, 0.0065386261471765145, 0.007021380840677843, 0.007435223751523479, 0.007779152867513715,
    #      0.008046724883868034, 0.008294120090597202, 0.008535229469085429, 0.008814563692978275, 0.009185456424947145,
    #      0.009708124355517132, 0.010449225373975225, 0.011480772284713654, 0.012869905662723773, 0.01465009477815408,
    #      0.016831968705570816, 0.0207038750779095, 0.058940312137735144, 0.5087192144171118, 1.0, 0.6166200097213604,
    #      0.26366256232476837, 0.07355497542954069, 0.019370175758140883, 0.009126744012311406, 0.005848566844176276,
    #      0.0041100117127384145, 0.0029871591741347796, 0.0022373760956846435, 0.0017715159454957101,
    #      0.0014454440630578874, 0.0012315778993680618, 0.0010854818065505758, 0.0009800249104588213,
    #      0.0008996926360163795, 0.0008358208280607502, 0.0007834451965917642, 0.0007395138068489446,
    #      0.0007019817924989364, 0.0006693863297672623, 0.0006406476758180254, 0.0006149640426597352,
    #      0.0005917430759409874, 0.0005705496510440059, 0.0005510642028858956, 0.0005330499930084517,
    #      0.0005163282919975373, 0.000500760246010398, 0.00048623406840327956, 0.00047265626784328765,
    #      0.00045994581525927433, 0.00044803037977921527, 0.000436843978126538, 0.0004263255612983882,
    #      0.00041702053536853926, 0.0004076376175539419, 0.0003987657264050876, 0.000390358190733434,
    #      0.00038237160417707984, 0.00037476576200815123, 0.00036750358552729774, 0.00036055102395896347,
    #      0.0003538769314655675, 0.00034745292156082894, 0.00034125320373217373, 0.00033525440814967893,
    #      0.0003294354044414357, 0.00032377712001918895, 0.00031826236260967827, 0.000312875650674516,
    #      0.0003076030544112346, 0.00030243204910204117, 0.0002973513817614045, 0.0002923509513518205,
    #      0.0002874217022947866, 0.00028255553059575617, 0.00027774520161560754, 0.0002729842783410281,
    #      0.0002685607030214653, 0.00026387988350864186]))

    inp = [0.0008996926360163795, 0.0008358208280607502, 0.0007834451965917642, 0.0007395138068489446,
           0.0007019817924989364, 0.0006693863297672623, 0.0006406476758180254, 0.0006149640426597352,
           0.0005163282919975373, 0.000500760246010398
           ]
    m = np.max(inp)
    for i in range(len(inp)):
        inp[i] = inp[i] / m
    print(inp)
    NN = NN(10, 4, 1, 2)
    print(NN.weights)
    print(NN.compute_output(inp))
    print(NN.compute_output(inp))
    print(NN.compute_output(inp))
    print(NN.compute_output(inp))
    print(NN.compute_output(inp))
    print(NN.compute_output(inp))
    print(NN.compute_output(inp))
