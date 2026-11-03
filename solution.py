import random
import sys
import time

from game_env import GameEnv
from game_state import GameState

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 3 "CrystalRover" Support Code

Last updated 16/10/2026
"""
# name: Juneyoung Jeong
# student number: 47026313
# github username: Yuwol6


class Solver:

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        #
        # TODO: Define any class instance variables you require (e.g. a dictionary mapping (state, action) to Q-value,
        #  or the weights of your linear Q-function) here.
        #
        pass

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5]

    # === Training rules (apply to both algorithms) ====================================================================
    #
    # Before training starts, the tester and autograder give self.game_env a training budget: a maximum number of
    # perform_action calls, and a lowest cumulative training reward. Once either runs out, perform_action raises
    # TrainingBudgetExhausted, which ends training - do not catch it. self.game_env.training_steps_used,
    # self.game_env.training_step_budget and self.game_env.training_reward_total let you plan around the budget.
    #
    # During training, perform_action must be called on either the initial state (self.game_env.get_init_state(), which
    # starts a new episode) or the state returned by your previous perform_action call.
    #
    # Your agent must learn from experience. The movement probabilities, action costs and penalties are hidden: each
    # testcase draws its own from its seed, and the environment keeps them private. Your agent may use the map (e.g.
    # distances, or what lies in an action's direction) and the rewards it receives, but must not read the
    # environment's private parameters or call its private methods, or plan with a transition model (e.g. value
    # iteration).
    #
    # Only some actions are valid on each tile (self.game_env.get_valid_actions(state)). An invalid action leaves the
    # state unchanged and still costs its action cost.

    # === Tabular Q-learning ===========================================================================================

    def ql_initialise(self):
        """
        Initialise any variables required before the start of tabular Q-learning.
        """
        #
        # TODO: Implement any initialisation for tabular Q-learning (e.g. creating your Q-table) here. This method is
        #  not timed, but it must not call perform_action (training belongs in ql_train_episode) and must finish
        #  within 30 seconds, or the phase scores zero.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        self.q_dict = {}
        self.alpha = 0.1  # learning rate
        self.epsilon = 0.2  # exploration rate

    def ql_train_episode(self):
        """
        Run one episode of tabular Q-learning.
        """
        #
        # TODO: Implement a single training episode here: start from self.game_env.get_init_state() and repeatedly
        #  choose an action (e.g. epsilon-greedy), call self.game_env.perform_action and update your Q-values, until
        #  the episode ends (a terminal state, or a step limit such as self.game_env.max_episode_steps).
        #
        # The tester and autograder call this method repeatedly until the training budget is exhausted, and check how
        # well your greedy policy performs at regular checkpoints in between.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        state = self.game_env.get_init_state()

        for _ in range(self.game_env.max_episode_steps):
            if self.game_env.is_terminal(state):
                break

            valid_actions = self.game_env.get_valid_actions(state)
            # exploration or exploitation
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                action = self.ql_select_action(state)

            # perform action
            next_state, reward, err_msg = self.game_env.perform_action(state, action)

            # save old Q-value
            old_q_value = self.q_dict.get((state, action), 0)

            if self.game_env.is_terminal(next_state):
                max_next_q = 0
            else:
                next_valid_actions = self.game_env.get_valid_actions(next_state)

                max_next_q = float("-inf")

                for next_action in next_valid_actions:
                    q_value = self.q_dict.get((next_state, next_action), 0)

                    if q_value > max_next_q:
                        max_next_q = q_value

            new_q_value = old_q_value + self.alpha * (
                reward + self.game_env.gamma * max_next_q - old_q_value
            )
            self.q_dict[(state, action)] = new_q_value

            state = next_state

    def ql_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned Q-values.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        #
        # TODO: Implement code to return the greedy action for the given state (based on your learned Q-values) here.
        #  This method is used to evaluate your agent (including during training), so it should not explore or update
        #  your Q-values.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        valid_actions = self.game_env.get_valid_actions(state)

        best_action = max(
            valid_actions, key=lambda action: self.q_dict.get((state, action), 0)
        )
        return best_action

    # === Feature-based Q-learning =====================================================================================

    def fq_initialise(self):
        """
        Initialise any variables required before the start of feature-based Q-learning.
        """
        #
        # TODO: Implement any initialisation for feature-based Q-learning (e.g. initial weights, or pre-computing
        #  information about the map that your features use) here. This method is not timed, so it is the place for
        #  precomputation, but it must not call perform_action (training belongs in fq_train_episode) and must finish
        #  within 30 seconds, or the phase scores zero.
        #
        # Your weights must start all 0 or all 1 (e.g. [0.0] * number of features) and be learned during training:
        #  starting from hand-picked weights scores zero for the phase.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        self.weights = [0.0] * 4
        self.fq_alpha = 0.01  # learning rate
        self.fq_epsilon = 0.2  # exploration rate

    def fq_get_features(self, state: GameState, action):
        """
        Return the feature vector of the given (state, action) pair, so that Q(s, a) = w . features(s, a).
        :param state: the current state
        :param action: an element of GameEnv.ACTIONS
        :return: a list (or 1-D numpy array) of at most 256 numbers, the same length for every (state, action) pair
        """
        #
        # TODO: Implement your feature design here. Features that describe the situation (e.g. distances, risks) rather
        #  than identify the exact state are what let your agent act well in states it never visited during training.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        # Get the intended movement direction

        match action[-1]:
            case "l":
                row_change, col_change = 0, -1
            case "r":
                row_change, col_change = 0, 1
            case "u":
                row_change, col_change = -1, 0
            case "d":
                row_change, col_change = 1, 0

        # Estimate the next position
        target_row = state.row + row_change
        target_col = state.col + col_change

        min_crystal_distance = float("inf")

        for i, (row, col) in enumerate(self.game_env.crystal_positions):
            if state.crystal_status[i] == 0:
                distance = abs(target_row - row) + abs(target_col - col)

                if distance < min_crystal_distance:
                    min_crystal_distance = distance

        # use 0 if all crystals are collected
        if min_crystal_distance == float("inf"):
            min_crystal_distance = 0

        # find distance to the nearest exit
        min_exit_distance = float("inf")

        for row, col in self.game_env.launch_positions:
            distance = abs(target_row - row) + abs(target_col - col)

            if distance < min_exit_distance:
                min_exit_distance = distance

        # check if the intended position is dangerous
        danger = 0

        if target_row < 0 or target_row >= self.game_env.n_rows:
            danger = 1
        elif target_col < 0 or target_col >= self.game_env.n_cols:
            danger = 1
        elif self.game_env.grid_data[target_row][target_col] in (
            self.game_env.LAVA_TILE,
            self.game_env.ROCK_TILE,
        ):
            danger = 1

        # add constant value for the agent to learn a baseline
        bias = 1

        # Normalise distances to approximately 0–1
        max_distance = self.game_env.n_rows + self.game_env.n_cols

        crystal_feature = min_crystal_distance / max_distance
        exit_feature = min_exit_distance / max_distance

        # Return the four values used to calculate the estimated q value
        return [crystal_feature, exit_feature, danger, bias]

    def fq_train_episode(self):
        """
        Run one episode of feature-based Q-learning (Q-learning with linear function approximation).
        """
        #
        # TODO: Implement a single training episode here, as for ql_train_episode, but updating the weights of your
        #  linear Q-function instead of a Q-table.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def fq_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned weights.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        #
        # TODO: Implement code to return the greedy action for the given state (based on your learned weights) here.
        #  This method is also evaluated from start positions your agent never trained from.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        valid_actions = self.game_env.get_valid_actions(state)

        best_action = None
        best_q_value = float("-inf")

        # find the action with the highest estimated q value
        for action in valid_actions:
            features = self.fq_get_features(state, action)

            # calculate q value using the weights and features
            q_value = 0

            for i in range(len(features)):
                q_value += self.weights[i] * features[i]

            # update the best action if a higher q value is found
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    #
