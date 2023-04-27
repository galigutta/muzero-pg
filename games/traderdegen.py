"""
This is a practice game setup where there are 5 slots where a trade can be placed.
Every period the trade generates a return, and the return deteriorates over time
The detrioration is stochastic. (The stochastic part may be implemented later)

At some point, the trade will start making negative returns, and the agent will
have to replace the trade with a new one. The agent can only replace one trade
per turn. The optimal replacement will be well before the trade turns negative,
or even before break-even.

The idea is to see how well the agent learns the optimal replacement strategy,
with and without the stochastic element.

There are questions of legal moves, where a trade cannot be placed in a slot that
is already occupied. Also if a trade slot is empty it cannot obviously be closed.
"""

import datetime
import math
import pathlib

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 1  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        
        


        ### Game
        self.n_slots = 5 # Number of slots for trades
        self.observation_shape = (1,1,self.n_slots*2) # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        # above is just a 1-d array of n_slots * 2, each indicating the last return of the slot, and the number of periods since the trade was opened
        self.action_space = list(range(self.n_slots + 2)) # Fixed list of all possible actions. You should only edit the length
        # The agent can open a trade (which goes into the first empty slot), close a specific trade, or do nothing: 1 + n_slots + 1
        self.players = list(range(1)) # List of players. You should only edit the length
        self.stacked_observations = 0 # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0 # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 12 # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 21 # Maximum number of moves if game is not finished before
        self.num_simulations = 21 # Number of future moves self-simulated
        self.discount = 0.85 # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 32  # Number of channels in the ResNet
        self.reduced_channels_reward = 32  # Number of channels in reward head
        self.reduced_channels_value = 32  # Number of channels in value head
        self.reduced_channels_policy = 32  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 30000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 200  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        # self.train_on_gpu = False

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.03  # Initial learning rate
        self.lr_decay_rate = 0.75  # Set it to 1 to use a constant learning rate
        # self.lr_decay_steps = 150000
        self.lr_decay_steps = self.training_steps / 10



        ### Replay Buffer
        self.replay_buffer_size = 1000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.05
            # return 0.25
        
        start_value, end_value, total_steps = 0.2, 0.01, self.training_steps
        decay_rate = math.log(end_value / start_value) / total_steps
        return start_value * math.exp(decay_rate * trained_steps)
        """
        return 0.05
    
class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = TraderDegen(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()


    def get_observation(self):
        return self.env.get_observation()    
    
    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        # input("Press enter to take a step ")

    def human_to_action(self):
        return self.env.human_to_action()
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(
            f"Enter the action (0) Hit, or (1) Stand for the player {self.to_play()}: "
        )
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter either (0) Hit or (1) Stand : ")
        return int(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        # actions = {
        #     "x": "Hit",
        #     1: "Stand",
        # }
        return self.env.action_to_string(action_number)


class TraderDegen:
    def __init__(self, seed):
        self.random = numpy.random.RandomState(seed)
        self.nslots = 5
        # age slots + return slots
        self.obs = [0]*self.nslots + [0]*self.nslots
        self.episode_len = 30
        self.step_count = 0
        self.step_rew = 0
        self.episode_rew = 0



    def to_play(self):
        return 0 # should always return 0 as there iy is always players turn
        # return 0 if self.player == 1 else 1 # should always return 0

    def reset(self):
        self.obs = [0]*self.nslots + [0]*self.nslots
        self.step_count = 0
        self.episode_rew = 0

        return self.get_observation()

        

    def get_observation(self):
        return [[self.obs]]

    def action_to_string(self, action_number):
        if action_number < self.nslots:
            return f'Close the trade in position {action_number}'
        elif action_number == self.nslots:
            return f'Do nothing - wait'
        elif action_number == self.nslots +1:
            return f'Open new trade'
    
    def legal_actions(self):
        opentrade = []
        # always an option to do nothing
        donothing = [self.nslots]
        # if there is space to open a trade, add that option
        if numpy.any(numpy.array(self.obs[:self.nslots]) == 0):
            opentrade = [self.nslots + 1]
        # pick the non zero ements for closing options
        close_options = list(numpy.nonzero(numpy.array(self.obs[:self.nslots]))[0])
        return close_options + donothing + opentrade
    
    def human_to_action(self):
        choice = input(
            f"Enter the action {self.nslots} Do nothing, or {self.nslots + 1} to open trade or index of trade to close: "
        )
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter a legal trade")
        return int(choice)

    def step(self, action):
        self.step_count += 1
        done = (self.step_count == self.episode_len)
        age_arr = numpy.array(self.obs[:self.nslots])
        ret_arr = numpy.array(self.obs[self.nslots:])

        # increment all not zeros by 1
        age_arr = numpy.where(age_arr == 0,0,age_arr+1)
        
        # if opening trade
        if action == self.nslots+1:
            for i, el in enumerate(age_arr):
                if el == 0:
                    age_arr[i] = 1
                    break
                    
        
        # if closing trade
        if action < self.nslots:
            age_arr[action] = 0
        
        # if no action
        if action == self.nslots:
            pass
        
        # generate an array or random normal returns with mean 1 and stdev of 0.1
        ret_arr = numpy.where(age_arr == 0,0,numpy.random.normal(1,0.1,self.nslots))
        
        # penalize by age**2; reduce the return by age**2 *0.03, where age is not 0
        ret_arr = ret_arr - numpy.where(age_arr == 0,0,(age_arr**2) *0.03)
        
        # calculate the reward for the step and episode:
        self.step_rew = numpy.sum(ret_arr)
        self.episode_rew += self.step_rew
        
        self.obs = list(age_arr) + list(ret_arr)
        
        return self.get_observation(), self.get_reward(done), done
                
    def get_reward(self, done):
        if not done:
            return self.step_rew
        else:
            return 0



    def render(self):
        print("Slots status: " + str(self.obs))
        print(f"step count: {self.step_count}, step reward {self.step_rew}, episode reward {self.episode_rew}" )
        

