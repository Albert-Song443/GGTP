import math
import time
import torch
import numpy as np
from planner_utils import *
from obs_adapter import *
from trajectory_tree_planner import TreePlanner
from ggtp_modules import GNNEncoder, GGTP_Decoder

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class GGTP_Planner(AbstractPlanner):
    """
    GGTP Planner: Game-theoretic planning with GNN encoder and CVAE decoder.
    """
    def __init__(self, model_path, device, num_samples=5):
        self._future_horizon = T  # [s] 
        self._step_interval = DT  # [s]
        self._N_points = int(T/DT)
        self._model_path = model_path
        self._device = device
        self._num_samples = num_samples  # Number of CVAE samples for prediction

    def name(self) -> str:
        return "GGTP Planner"
    
    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = GGTP_TreePlanner(
            self._device, 
            self._encoder, 
            self._decoder,
            num_samples=self._num_samples
        )

    def _initialize_model(self):
        model = torch.load(self._model_path, map_location=self._device)
        
        # GNN Encoder
        self._encoder = GNNEncoder(
            node_dim=11,
            edge_dim=4,
            dim=256,
            heads=4,
            layers=2
        )
        self._encoder.load_state_dict(model['encoder'])
        self._encoder.to(self._device)
        self._encoder.eval()
        
        # CVAE Decoder
        self._decoder = GGTP_Decoder(
            neighbors=10,
            max_time=8,
            max_branch=30,
            agent_dim=256,
            ego_plan_dim=256,
            latent_dim=32,
            horizon=80,
            traj_dim=3
        )
        self._decoder.load_state_dict(model['decoder'])
        self._decoder.to(self._device)
        self._decoder.eval()

    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

    def compute_planner_trajectory(self, current_input: PlannerInput):
        # Extract iteration, history, and traffic light
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state

        # Construct input features
        start_time = time.perf_counter()
        features = observation_adapter(
            history, 
            traffic_light_data, 
            self._map_api, 
            self._route_roadblock_ids, 
            self._device
        )

        # Get starting block
        starting_block = None
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
        
        # Get traffic light lanes
        traffic_light_lanes = []
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                traffic_light_lanes.append(lane_conn)

        # Tree policy planner
        try:
            plan = self._trajectory_planner.plan(
                iteration, 
                ego_state, 
                features, 
                starting_block, 
                self._route_roadblocks, 
                self._candidate_lane_edge_ids, 
                traffic_light_lanes, 
                observation
            )
        except Exception as e:
            print("Error in planning:", e)
            plan = np.zeros((self._N_points, 3))
            
        # Convert relative poses to absolute states and wrap in a trajectory object
        states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)
        print(f'Step {iteration+1} Planning time: {time.perf_counter() - start_time:.3f} s')

        return trajectory


class GGTP_TreePlanner(TreePlanner):
    """
    Extended TreePlanner that uses CVAE for multi-sample prediction.
    """
    def __init__(self, device, encoder, decoder, num_samples=5, **kwargs):
        super().__init__(device, encoder, decoder, **kwargs)
        self.num_samples = num_samples
    
    def predict(self, encoder_outputs, traj_inputs, agent_states, timesteps):
        """
        Override predict to generate multiple CVAE samples and average.
        """
        # Prepare ego trajectories
        ego_trajs = torch.zeros((self.n_candidates_max, self.horizon*10, 6)).to(self.device)
        for i, traj in enumerate(traj_inputs):
            ego_trajs[i, :len(traj)] = traj[..., :6].float()
        ego_trajs = ego_trajs.unsqueeze(0)
        
        # Multiple samples for robustness
        all_samples = []
        all_scores = []
        
        for _ in range(self.num_samples):
            agent_trajs, scores, _ = self.decoder(
                encoder_outputs, 
                ego_trajs, 
                agent_states, 
                timesteps
            )
            all_samples.append(agent_trajs)
            all_scores.append(scores)
        
        # Average predictions and scores
        agent_trajs = torch.stack(all_samples).mean(0)
        scores = torch.stack(all_scores).mean(0)
        
        return agent_trajs, scores

