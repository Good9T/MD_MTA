from dataclasses import dataclass
import torch
from MDOVRP_Problem import get_random_problems, augment_x_y_by_8


@dataclass
class Reset_State:
    depot_x_y: torch.Tensor = None
    # shape: (batch, depot, 2)
    customer_x_y: torch.Tensor = None
    # shape: (batch, customer, 2)
    customer_demand: torch.Tensor = None
    # shape: (batch, customer)
    depot_size: torch.int = None
    mt_size: torch.int = None
    customer_size: torch.int = None
    capacity: torch.int = None


@dataclass
class Step_State:
    batch_idx: torch.Tensor = None
    mt_idx: torch.Tensor = None
    selected_count: int = None
    current_node: torch.Tensor = None
    finished: torch.Tensor = None
    # shape: (batch, mt)
    mask: torch.Tensor = None
    # shape: (batch, mt, node)


class MDOVRPEnvTrain:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.customer_size = self.env_params['customer_size']
        self.mt_size = self.env_params['mt_size']
        self.depot_size = self.env_params['depot_size']
        self.capacity = self.env_params['capacity']
        self.demand_min = self.env_params['demand_min']
        self.demand_max = self.env_params['demand_max']


        self.batch_idx = None
        self.mt_idx = None
        # shape: (batch, mt)
        self.batch_size = None
        self.depot_customer_x_y = None
        # shape: (batch, node, 2)
        self.depot_customer_demand = None
        # shape: (batch, node)

        self.selected_count = None
        self.current_node = None
        # shape: (batch, mt)
        self.selected_node_list = None
        # shape: (batch, mt, 0~)

        self.at_the_depot = None
        self.last_at_the_depot = None
        # shape: (batch, mt)
        self.load = None
        # shape: (batch, mt)
        self.visited_flag = None
        # shape: (batch, mt, node)
        self.mask = None
        # shape: (batch, mt, node)
        self.finished = None
        # shape: (batch, mt)

        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size):

        self.batch_size = batch_size
        depot_x_y, customer_x_y, customer_demand, _ = get_random_problems(
            batch_size=batch_size, depot_size=self.depot_size, customer_size=self.customer_size, 
            capacity=self.capacity, demand_min=self.demand_min, demand_max=self.demand_max, aug_type=None)
        depot_x_y = depot_x_y.to('cuda:0')
        customer_x_y = customer_x_y.to('cuda:0')
        customer_demand = customer_demand.to('cuda:0')

        self.depot_customer_x_y = torch.cat((depot_x_y, customer_x_y), dim=1)
        # shape: (batch_size, depot_size + customer_size, 2)
        depot_demand = torch.zeros(size=(self.batch_size, self.depot_size))
        # shape: (batch_size, depot_size)
        self.depot_customer_demand = torch.cat((depot_demand, customer_demand), dim=1)
        # shape: (batch_size, depot_size + customer_size)
        self.batch_idx = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.mt_size)
        self.mt_idx = torch.arange(self.mt_size)[None, :].expand(self.batch_size, self.mt_size)

        self.reset_state.depot_x_y = depot_x_y
        self.reset_state.customer_x_y = customer_x_y
        self.reset_state.customer_demand = customer_demand
        self.reset_state.depot_size = self.depot_size
        self.reset_state.customer_size = self.customer_size
        self.reset_state.mt_size = self.mt_size
        self.reset_state.capacity = self.capacity

        self.step_state.batch_idx = self.batch_idx
        self.step_state.mt_idx = self.mt_idx

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, mt)
        self.selected_node_list = torch.zeros((self.batch_size, self.mt_size, 0), dtype=torch.long)
        # shape: (batch, mt, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        # shape: (batch, mt)
        self.last_at_the_depot = torch.ones(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        # shape: (batch, mt)
        self.load = torch.ones(size=(self.batch_size, self.mt_size))
        # shape: (batch, mt)
        self.visited_flag = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size + self.customer_size))
        # shape: (batch, mt, node)
        self.mask = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size + self.customer_size))
        # shape: (batch, mt, node)
        self.finished = torch.zeros(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        # shape: (batch, mt)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.mask = self.mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, mt)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, mt, 0~)
        self.at_the_depot = (selected < self.depot_size)
        demand_list = self.depot_customer_demand[:, None, :].expand(self.batch_size, self.mt_size, -1)
        # shape: (batch, mt, node)
        index_to_gather = selected[:, :, None]
        # shape: (batch, mt, 1)
        selected_demand = demand_list.gather(dim=2, index=index_to_gather).squeeze(dim=2)
        # shape: (batch, mt)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill at the depot
        self.visited_flag[self.batch_idx, self.mt_idx, selected] = float('-inf')
        # shape: (batch, mt, node)
        self.visited_flag[:, :, 0:self.depot_size] = 0
        self.visited_flag[:, :, 0:self.depot_size][self.last_at_the_depot + self.at_the_depot] = float('-inf')
        self.last_at_the_depot = self.at_the_depot
        new_finished = (self.visited_flag[:, :, self.depot_size:] == float('-inf')).all(dim=2)
        # shape: (batch, mt)
        self.finished = self.finished + new_finished
        self.mask = self.visited_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, mt, node)
        self.mask[demand_too_large] = float('-inf')
        # shape: (batch, mt)

        # shape: (batch, mt)
        self.mask[:, :, 0:self.depot_size][self.finished] = 0  # do not mask depot for finished episode.
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.mask = self.mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        if done:
            reward = -self.get_travel_distance()
        else:
            reward = None

        return self.step_state, reward, done

    def get_travel_distance(self):
        index_to_gather = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, mt, selected_list, 2)
        all_x_y = self.depot_customer_x_y[:, None, :, :].expand(-1, self.mt_size, -1, -1)
        # shape: (batch, mt, node, 2)
        seq_ordered = all_x_y.gather(dim=2, index=index_to_gather)
        depot_ordered = self.selected_node_list < self.depot_size
        # shape: (batch, mt, selected_list, 2)
        depot_rolled = depot_ordered.roll(dims=2, shifts=-1)
        depot_final = depot_rolled
        seq_rolled = seq_ordered.roll(dims=2, shifts=-1)
        segment_lengths = ((seq_ordered - seq_rolled) ** 2).sum(3).sqrt()
        # shape: (batch, mt, selected_list)
        segment_lengths[depot_final] = 0
        travel_distances = segment_lengths.sum(2)
        # shape: (batch, mt)
        return travel_distances
