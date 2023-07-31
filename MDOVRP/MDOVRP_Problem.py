import torch
import numpy as np
import pandas as pd
import copy


def get_random_problems(batch_size, depot_size, customer_size, capacity=50, demand_min=1, demand_max=10,
                        aug_type=None):
    node_size = depot_size + customer_size

    depot_x_y = torch.rand(size=(batch_size, depot_size, 2))
    # shape: (batch, depot_size, 2)
    customer_x_y = torch.rand(size=(batch_size, customer_size, 2))
    # shape: (batch, customer_size, 2)
    customer_demand = torch.randint(demand_min, demand_max, size=(batch_size, customer_size)) / capacity
    depot_x_y, customer_x_y, customer_demand, aug_factor = aug(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand)

    return depot_x_y, customer_x_y, customer_demand, aug_factor


def get_dataset_problem(load_path, batch_size, aug_type='d'):

    filename = load_path
    data = pd.read_csv(filename, sep=',', header=None)
    data = data.to_numpy()
    depot_size = int(data[0][0])
    customer_size = int(data[0][1])
    capacity = int(data[0][2])
    scale = int(data[0][3])
    depot_xyd = data[1:depot_size + 1]
    customer_xyd = data[depot_size + 1:depot_size + customer_size + 1]
    full_node = data[1:depot_size + customer_size + 1]
    for i in range(len(depot_xyd)):
        depot_x_y = torch.FloatTensor(depot_xyd[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
            [depot_x_y, torch.FloatTensor(depot_xyd[i][0:2]).unsqueeze(0)], dim=0)
    for i in range(len(customer_xyd)):
        customer_x_y = torch.FloatTensor(customer_xyd[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
            [customer_x_y, torch.FloatTensor(customer_xyd[i][0:2]).unsqueeze(0)], dim=0)
        customer_demand = torch.FloatTensor(customer_xyd[i][2:3]) if i == 0 else torch.cat(
            [customer_demand, torch.FloatTensor(customer_xyd[i][2:3])], dim=0)
    customer_demand = customer_demand / capacity
    depot_x_y = depot_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    customer_x_y = customer_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    customer_demand = customer_demand.unsqueeze(0).repeat(batch_size, 1)
    depot_x_y, customer_x_y, customer_demand, aug_factor = aug(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand)
    data = {'depot_x_y': depot_x_y.numpy().tolist(), 'customer_x_y': customer_x_y.numpy().tolist(),
            'customer_demand': customer_demand.numpy().tolist(), 'capacity': capacity, 'full_node': full_node,
            'scale': scale, 'aug_factor': aug_factor}
    return depot_x_y, customer_x_y, customer_demand, depot_size, customer_size, capacity, data, aug_factor


def get_1_random_problems(batch_size, depot_size, customer_size, capacity=50, demand_min=1, demand_max=10,
                          aug_type=None):
    depot_x_y = torch.rand(size=(depot_size, 2)).repeat(batch_size, 1, 1)

    customer_x_y = torch.rand(size=(customer_size, 2)).repeat(batch_size, 1, 1)

    customer_demand = torch.randint(demand_min, demand_max, size=(1, customer_size)).repeat(batch_size, 1) / capacity

    depot_x_y, customer_x_y, customer_demand, aug_factor = aug(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand)
    return depot_x_y, customer_x_y, customer_demand, aug_factor


def aug(aug_type, depot_size, depot_x_y, customer_x_y, customer_demand):
    aug_factor = 1
    if aug_type == 'd':
        aug_factor = depot_size + 7
        depot_x_y = augment_x_y_by_d(depot_x_y, depot_size)
        customer_x_y = augment_x_y_by_d(customer_x_y, depot_size)

    elif aug_type == '8':
        aug_factor = 8
        depot_x_y = augment_x_y_by_8(depot_x_y)
        customer_x_y = augment_x_y_by_8(customer_x_y)
    customer_demand = customer_demand.repeat(aug_factor, 1)

    return depot_x_y, customer_x_y, customer_demand, aug_factor

def augment_x_y_by_8(x_y):
    # shape: (batch, N, 2)

    x = x_y[:, :, [0]]
    y = x_y[:, :, [1]]
    # shape: (batch, N, 1)

    data1 = torch.cat((x, y), dim=2)
    data2 = torch.cat((1 - x, y), dim=2)
    data3 = torch.cat((x, 1 - y), dim=2)
    data4 = torch.cat((1 - x, 1 - y), dim=2)
    data5 = torch.cat((y, x), dim=2)
    data6 = torch.cat((1 - y, x), dim=2)
    data7 = torch.cat((y, 1 - x), dim=2)
    data8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_x_y = torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=0)
    # shape: (8 * batch_size, N, 2)
    return aug_x_y


def augment_x_y_by_d(x_y, depot_size):
    # shape: (batch, N, 2)

    x = x_y[:, :, [0]]
    y = x_y[:, :, [1]]
    # shape: (batch, N, 1)

    data1 = torch.cat((x, y), dim=2)
    data2 = torch.cat((1 - x, y), dim=2)
    data3 = torch.cat((x, 1 - y), dim=2)
    data4 = torch.cat((1 - x, 1 - y), dim=2)
    data5 = torch.cat((y, x), dim=2)
    data6 = torch.cat((1 - y, x), dim=2)
    data7 = torch.cat((y, 1 - x), dim=2)
    data8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_x_y = torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=0)
    # shape: (8 * batch_size, N, 2)
    for i in range(depot_size-1):
        x_y_temp = copy.deepcopy(x_y)
        x_y_temp[:, 0] = x_y_temp[:, i+1]

        x_y_temp[:, i+1] = x_y[:, 0]
        aug_x_y = torch.cat((aug_x_y, x_y_temp), dim=0)
    return aug_x_y


