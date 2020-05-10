from maze_env import Maze
from RL_brain import DQN


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.memorize(observation, action, reward, observation_)

            if (step > 200) and (step % 3 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DQN(env.n_actions,
             memory_size=300,
             reward_decay=0.9,
             e_greedy=0.9,
             replace_target_iter=100
             )
    env.after(100, run_maze)
    env.mainloop()
