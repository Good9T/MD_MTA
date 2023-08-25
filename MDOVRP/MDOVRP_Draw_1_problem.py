import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import copy


def Draw_1_Problem(data, solution, result_folder):
    depot_size = len(data['depot_x_y'])
    customer_size = len(data['customer_x_y'])
    node_size = depot_size + customer_size
    node_size = depot_size + customer_size
    full_node = copy.deepcopy(data['full_node'])

    depot_dict = {k: v for k, v in zip(range(0, depot_size), data['depot_x_y'])}
    customer_dict = {k: v for k, v in zip(range(depot_size, node_size), data['customer_x_y'])}
    full_dict = {}
    full_dict.update(depot_dict)
    full_dict.update(customer_dict)
    solution_list = solution.squeeze(0).cpu().numpy().tolist()
    node = []
    for i in solution_list:
        node.append(full_node[i])

    # plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (20, 15)
    plt.title(u'problem')
    p1 = []
    p2 = []
    for l1 in full_dict.values():
        p1.append(l1[0])
        p2.append(l1[1])
    plt.plot(p1[:depot_size], p2[:depot_size], 'g*', ms=40, label='depot')
    plt.plot(p1[depot_size:], p2[depot_size:], 'ko', ms=25, label='customer')
    plt.grid(True)

    plt.savefig('{}/problem.jpg'.format(result_folder))
    plt.show()

    # plot routes

    plt.rcParams['axes.unicode_minus'] = False
    plt.title(u'routes')
    p1 = []
    p2 = []
    for l2 in full_dict.values():
        p1.append(l2[0])
        p2.append(l2[1])
    plt.plot(p1[:depot_size], p2[:depot_size], 'g*', ms=30, label='depot')
    plt.plot(p1[depot_size:], p2[depot_size:], 'ko', ms=20, label='customer')
    plt.grid(True)
    plt.legend(loc='lower left')
    for i in range(0, len(node) - 1):
        start = node[i]
        end = node[i + 1]
        if end[2] == 0:
            plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], width=0.002, length_includes_head=True,
                      color='b')

    plt.savefig('{}/routes.jpg'.format(result_folder))
    plt.show()

