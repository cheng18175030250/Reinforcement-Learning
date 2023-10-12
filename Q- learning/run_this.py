from maze_env import Maze
from RL_brain import QLearningTable

def update():
	for episode in range(150):
		observation = env.reset()
		print(episode)
		while True:
			env.render()
			# 2. 基于新状态再通过ε-greedy策略选择动作
			# ϵ−贪心策略是一种常用的策略。其表示在智能体做决策时，有一很小的正数 ϵ ( < 1 ) 的概率随机选择未知的一个动作，
			# 剩下 1 − ϵ 的概率选择已有动过中动作价值最大的动作。
			action = RL.choose_action(str(observation))
			# print("observation: {}".format(observation))
			observation_, reward, done = env.step(action)

			# 1. 取最大奖赏的动作，更新值函数
			RL.learn(str(observation), action, reward, str(observation_))
			# print(RL.q_table)
			observation = observation_
			if done:
				break
	print('game over')
	env.destroy()

if __name__ == '__main__':
	env = Maze()
	# print("env.n_actions: {}".format(env.n_actions))
	RL = QLearningTable(actions=list(range(env.n_actions)))

	env.after(100, update)
	env.mainloop()