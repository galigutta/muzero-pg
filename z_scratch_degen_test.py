import importlib

class MuZero:
    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err


muzero = MuZero("traderdegen")
env = muzero.Game()
env.reset()
env.render()
print("legal actions: ", env.legal_actions())
# env.human_to_action()

# obs = env.get_observation()
# print(obs)

done = False
while not done:
    action = env.human_to_action()
    observation, reward, done = env.step(action)
    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
    env.render()