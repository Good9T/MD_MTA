from dataclasses import dataclass
import torch
from MDVRP_Problem import get_random_problems, augment_x_y_by_8


@dataclass
class Reset_State:
    depot_x_y: torch.Tensor = None
    # shape: (batch_size, depot_size, 2)
    customer_x_y: torch.Tensor = None
    # shape: (batch_size, customer_size, 2)
    customer_demand: torch.Tensor = None
    # shape: (batch_size, customer_size)
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
    # shape: (batch_size, mt_size)
    mask: torch.Tensor = None
    # shape: (batch_size, mt_size, node)


class MDVRPEnvTrain:
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
        # shape: (batch_size, mt_size)
        self.batch_size = None
        self.depot_customer_x_y = None
        # shape: (batch_size, node, 2)
        self.depot_customer_demand = None
        # shape: (batch_size, node)

        self.selected_count = None
        self.current_node = None
        # shape: (batch_size, mt_size)
        self.selected_node_list = None
        # shape: (batch_size, mt_size, 0~)

        self.departure_depot = None
        self.at_the_depot = None
        self.last_at_the_depot = None
        # shape: (batch_size, mt_size)
        self.load = None
        # shape: (batch_size, mt_size)
        self.visited_flag = None
        # shape: (batch_size, mt_size, node)
        self.mask = None
        # shape: (batch_size, mt_size, node)
        self.finished = None
        # shape: (batch_size, mt_size)

        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, aug_factor=1):

        self.batch_size = batch_size
        depot_x_y, customer_x_y, customer_demand, _ = get_random_problems(
            batch_size=batch_size, depot_size=self.depot_size, customer_size=self.customer_size, 
            capacity=self.capacity, demand_min=self.demand_min, demand_max=self.demand_max, aug_type=None)
        depot_x_y = depot_x_y.to('cuda:0')
        customer_x_y = customer_x_y.to('cuda:0')
        customer_demand = customer_demand.to('cuda:0')

        self.depot_customer_x_y = torch.cat((depot_x_y, customer_x_y), dim=1)
        # shape: (batch_size, node, 2)
        depot_demand = torch.zeros(size=(self.batch_size, self.depot_size))
        # shape: (batch_size, depot_size)
        self.depot_customer_demand = torch.cat((depot_demand, customer_demand), dim=1)
        # shape: (batch_size, node)
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
        # shape: (batch_size, mt_size)
        self.selected_node_list = torch.zeros((self.batch_size, self.mt_size, 0), dtype=torch.long)
        # shape: (batch_size, mt_size, 0~)
        self.departure_depot = torch.zeros(size=(self.batch_size, self.mt_size), dtype=torch.long)
        # shape: (batch_size, mt_size)
        self.at_the_depot = torch.ones(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        # shape: (batch_size, mt_size)
        self.last_at_the_depot = torch.ones(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        # shape: (batch_size, mt_size)
        self.load = torch.ones(size=(self.batch_size, self.mt_size))
        # shape: (batch_size, mt_size)
        self.visited_flag = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size + self.customer_size))
        # shape: (batch_size, mt_size, node)
        self.mask = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size + self.customer_size))
        # shape: (batch_size, mt_size, node)
        self.finished = torch.zeros(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        # shape: (batch_size, mt_size)

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
        # shape: (batch_size, mt_size)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch_size, mt_size, 0~)
        self.at_the_depot = (selected < self.depot_size)
        self.departure_depot[self.at_the_depot] = self.current_node[self.at_the_depot]
        demand_list = self.depot_customer_demand[:, None, :].expand(self.batch_size, self.mt_size, -1)
        # shape: (batch_size, mt_size, node)
        index_to_gather = selected[:, :, None]
        # shape: (batch_size, mt_size, 1)
        selected_demand = demand_list.gather(dim=2, index=index_to_gather).squeeze(dim=2)
        # shape: (batch_size, mt_size)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill at the depot
        self.visited_flag[self.batch_idx, self.mt_idx, selected] = float('-inf')
        # shape: (batch_size, mt_size, node_size)
        self.visited_flag[:, :, 0:self.depot_size][self.at_the_depot] = 0
        self.visited_flag[:, :, 0:self.depot_size][self.last_at_the_depot] = float('-inf')

        self.visited_flag[:, :, 0:self.depot_size][self.last_at_the_depot + ~self.at_the_depot] = float('-inf')
        self.visited_flag[self.batch_idx, self.mt_idx, self.departure_depot] = 0
        self.visited_flag[self.batch_idx, self.mt_idx, selected] = float('-inf')
        self.last_at_the_depot = self.at_the_depot
        self.mask = self.visited_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch_size, mt_size, node_size)
        self.mask[demand_too_large] = float('-inf')
        # shape: (batch_size, mt_size)

        new_finished = (self.visited_flag[:, :, self.depot_size:] == float('-inf')).all(dim=2)
        # shape: (batch_size, mt_size)
        self.finished = self.finished + new_finished
        self.mask[:, :, 0:self.depot_size][self.finished] = 0  # do not mask depot for finished episode.
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.mask = self.mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        if done:
            reward = -self.get_travel_distance()
            self.step_state.result_node_list = self.selected_node_list
        else:
            reward = None

        return self.step_state, reward, done

    def get_travel_distance(self):
        index_to_gather = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch_size, mt_size, selected_list, 2)
        all_x_y = self.depot_customer_x_y[:, None, :, :].expand(-1, self.mt_size, -1, -1)
        # shape: (batch_size, mt_size, node, 2)
        seq_ordered = all_x_y.gather(dim=2, index=index_to_gather)
        depot_ordered = self.selected_node_list < self.depot_size
        # shape: (batch_size, mt_size, selected_list, 2)
        depot_rolled = depot_ordered.roll(dims=2, shifts=-1)
        depot_final = depot_ordered * depot_rolled
        seq_rolled = seq_ordered.roll(dims=2, shifts=-1)
        segment_lengths = ((seq_ordered - seq_rolled) ** 2).sum(3).sqrt()
        # shape: (batch_size, mt_size, selected_list)
        segment_lengths[depot_final] = 0
        travel_distances = segment_lengths.sum(2)
        # shape: (batch_size, mt_size)
        return travel_distances
