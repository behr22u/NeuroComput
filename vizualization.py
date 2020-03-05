import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

class Vizualization:
    # dataset : numpy array containing N two-dimensional vectors
    # graph_shape : shape of the som (by default 10 *10 grid)
    def __init__(self,
                 dataset,
                 graph_shape = (10,10),
                 show_current_input = True,
                 show_dataset = True):

        self.show_current_input = show_current_input
        self.show_dataset = show_dataset
        self.title = ""

        # graph for SOM
        self.graph = nx.grid_graph(dim=[*graph_shape],periodic = False)
        self.nodes_indices = sorted(self.graph.nodes())
        norm_y = graph_shape[0]-1
        norm_x = graph_shape[1]-1
        self.pos_nodes = dict(((y,x), (y/norm_y, x/norm_x)) for y,x in self.nodes_indices)

        # graph for all data
        self.dataset = dataset
        self.data_graph = nx.Graph()
        self.data_graph.add_nodes_from(np.arange(len(self.dataset)))
        self.data_nodes_indices = sorted(self.data_graph.nodes())
        self.data_pos_nodes = dict((idx,(dataset[idx,0], dataset[idx,1])) for idx in self.data_nodes_indices)

        # graph for current input data
        self.current_data_graph = nx.Graph()
        self.current_data_graph.add_node(0)
        self.current_data_nodes_indices = sorted(self.current_data_graph.nodes())
        self.current_data_pos_node = {0: (0,0)}

        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])
        self.ax.clear()
        self.draw_map()
        plt.show()

    def weights_to_pos (self, weights):
        for idx in self.nodes_indices :
            self.pos_nodes[idx] = tuple(weights[idx])

    def draw_map (self):
        self.ax.clear()
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])

        nx.draw_networkx_nodes(
            self.data_graph,
            ax =self.ax,
            pos=self.data_pos_nodes,
            nodelist=self.data_nodes_indices,
            node_color='red',
            node_size = 300,
            node_shape = "s",
            alpha = 1.0)

        nx.draw_networkx(self.graph,
                         ax = self.ax,
                         with_labels=False,
                         pos =  self.pos_nodes,
                         nodelist = self.nodes_indices,
                         node_color= 'black',
                         node_size = 200)

        nx.draw_networkx_nodes(
            self.current_data_graph,
            ax =self.ax,
            pos=self.current_data_pos_node,
            nodelist=self.current_data_nodes_indices,
            node_color='green',
            node_size = 800,
            node_shape = "s",
            alpha = 0.5)

        self.ax.set_title(self.title, fontweight="bold")

    # update : function to call to update the vizualization
    # current_input_vector : two dimensional input_vector
    # som_tuned_values : numpy array of shape (nb_neurons_y, nb_neurons_x, 2)
    # plot_title : string that contains the title of the plot
    def update(self,
               current_input_vector,
               som_tuned_values,
               plot_title = ""):

        self.current_data_pos_node[0] = tuple(current_input_vector)
        self.title = plot_title
        self.weights_to_pos(som_tuned_values)
        self.draw_map()
        self.fig.canvas.start_event_loop(0.001)
