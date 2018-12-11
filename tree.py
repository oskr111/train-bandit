"""
python3
バンディット問題を強化学習するプログラム

スロットマシンは２つ存在していて、それぞれの確率は[0.4, 0.6]
当たると報酬が１もらえる

方策はgreedyとε-greedyとε減衰の３つ(このコードにはgreedyしか実装されていない)

残りの二つを実装するのが課題　(このコードから編集するなら)
"""

import numpy as np
import matplotlib.pyplot as plt
import random


# バンディットタスク
class Bandit():
    def __init__(self, p):
        # バンディットの設定
        self.probability = p
        # スタート地点
        self.start = 0
        # ゴール地点
        self.goal = len(self.probability)

    # 報酬を評価
    def get_reward(self, current_state, action):
        # 受け取るアクションは0か1の2値
        # アタリなら１を返す
        if random.random() <= self.probability[current_state, action]:
            return 1
        # 外れなら0を返す
        else:
            return 0

    # 当たる確率を返す
    def get_probability(self, current_state, action):
        return self.probability[current_state][action]

    # 状態の数を返す
    def get_num_state(self):
        return len(self.probability)

    # 行動の数を返す
    def get_num_action(self):
        return len(self.probability[0])

    # スタート地点の場所を返す(初期化用)
    def get_start(self):
        return self.start

    # ゴール地点の場所を返す
    def get_goal_state(self):
        return self.goal

    # 行動を受け取り、次状態を返す
    def get_next_state(self, current_state, current_action):
        if self.get_goal_state() != 1 and current_state == 0:
            return current_state + 1 + current_action
        return self.goal


# Q学習のクラス
class Q_learning():
    # 学習率、割引率、状態数、行動数を定義する
    def __init__(self, learning_rate=0.03, discount_rate=0.9, num_state=None, num_action=None):
        self.learning_rate = learning_rate  # 学習率
        self.discount_rate = discount_rate  # 割引率
        self.num_state = num_state  # 状態数
        self.num_action = num_action  # 行動数
        # Qテーブルを初期化
        self.Q = np.zeros((self.num_state+1, self.num_action))

    # Q値の更新
    # 現状態、選択した行動、得た報酬、次状態を受け取って更新する
    def update_Q(self, current_state, current_action, reward, next_state, next_action):
        # TD誤差の計算
        TD_error = (reward
                    + self.discount_rate
                    * max(self.Q[next_state])
                    - self.Q[current_state, current_action])
        # Q値の更新
        self.Q[current_state, current_action] += self.learning_rate * TD_error

    # Q値の初期化
    def init_params(self):
        self.Q = np.zeros((self.num_state+1, self.num_action))

    # Q値を返す
    def get_Q(self):
        return self.Q

# Sarsaのクラス
class Sarsa(Q_learning):
    def __init__(self, learning_rate=0.03, discount_rate=0.9, num_state=None, num_action=None):
        super().__init__(learning_rate, discount_rate, num_state, num_action)
    # Q値の更新
    # 現状態、選択した行動、得た報酬、次状態を受け取って更新する
    def update_Q(self, current_state, current_action, reward, next_state, next_action):
        # TD誤差の計算
        #next_action =
        TD_error = (reward
                    + self.discount_rate
                    * self.Q[next_state, next_action]
                    - self.Q[current_state, current_action])
                    # Q値の更新
        self.Q[current_state, current_action] += self.learning_rate * TD_error



# 方策クラス
class Greedy():  # greedy方策
    # 行動価値を受け取って行動番号を返す
    def serect_action(self, value, current_state):
        return np.argmax(value[current_state])


class PS(object):
    def __init__(self, r):
            self.r = r
    def serect_action(self, value, current_state):
        idx = np.where(value[current_state] == np.max(value[current_state]))
        action = random.choice(idx[0])
        if value[current_state][action] > self.r :
            return action
        return np.random.randint(0, len(value[current_state]))
    def get_tau(self):
        return 0
    def init_tau(self):
        pass

class RS(PS):
    def __init__(self, r_state):
        self.r_state = r_state      # 基準値
    def serect_action(self,value,current_state):
       # もっともtauの中で大きいものの場所
        #tau_before = np.where(self.tau_state == np.max(self.tau_state))
        #tau0 = random.choice(tau_before[0])
        tau = self.tau_state
        RS=[0,0,0,0]
        # RS値そのもの
        for i in range(4):
            RS[i] = tau[current_state][i]*(value[current_state] - self.r_state)
        idx = np.where(RS == np.max(RS))
        action = random.choice(idx[0])
        self.tau_state[current_state][action] += 1
        return action
    def init_tau(self):
        self.tau_state = np.zeros((5+1, 4))
    def get_tau(self):
        return self.tau_state


# エージェントクラス
class Agent():
    def __init__(self, value_func="Q_learning", policy = "RS", learning_rate=0.05, discount_rate=0.9, n_state=None, n_action=None, n_bandit=None):
        # 価値更新方法の選択
        if value_func == "Q_learning":
            self.value_func = Q_learning(num_state=n_state, num_action=n_action)
        elif value_func == "Sarsa":
            self.value_func = Sarsa(num_state=n_state, num_action=n_action)
        else:
            print("error:価値関数候補が見つかりませんでした")
            sys.exit()

        # 方策の選択
        if policy == "PS":
            self.policy = PS(0.5)
        elif policy == "RS":
            self.policy = RS(0.5)
        else:
            print("error:方策候補が見つかりませんでした")
            sys.exit()

    # パラメータ更新(基本呼び出し)
    def update(self, current_state, current_action, reward, next_state, next_action):
        self.value_func.update_Q(current_state, current_action, reward, next_state, next_action)

    # 行動選択(基本呼び出し)
    def serect_action(self, current_state):
        return self.policy.serect_action(self.value_func.get_Q(), current_state)

    # 行動価値の表示
    def print_value(self):
        print(self.value_func.get_Q())

    # 所持パラメータの初期化
    def init_params(self):
        self.value_func.init_params()

    def init_tau(self):
        self.policy.init_tau()

    def get_tau(self):
        print(self.policy.get_tau())


# メイン関数
def main():
    # ハイパーパラメータ等の設定
    task = Bandit(p = np.array([[0.4, 0.5, 0.6, 0.7], [0.9, 0.8, 0.1, 0.2], [0.7, 0.1, 0.3, 0.6], [0.6, 0.4, 0.1, 0.3], [0.5, 0.2, 0.3, 0.1]]))  # タスク定義
    #task = Bandit(p = np.asarray([[0.1, 0.3, 0.5, 0.9]]))
    max = 0.4 * 0.9
    #max = 0.9

    SIMULATION_TIMES = 100     # シミュレーション回数
    EPISODE_TIMES = 1000  # エピソード回数

    agent = {}
    agent[0] = Agent(value_func="Q_learning", policy="PS", n_state=task.get_num_state(), n_action=task.get_num_action())
    agent[1] = Agent(value_func="Q_learning", policy="RS", n_state=task.get_num_state(), n_action=task.get_num_action())
    #agent[2] = Agent(value_func="Sarsa", policy="PS", n_state=task.get_num_state(), n_action=task.get_num_action())
    #agent[3] = Agent(value_func="Sarsa", policy="RS", n_state=task.get_num_state(), n_action=task.get_num_action())
    # print(task.get_num_state())
    # print(task.get_num_action())

    # グラフ記述用の記録
    reward_graph = np.zeros((len(agent), EPISODE_TIMES))
    # accuracy = np.zeros((len(agent), EPISODE_TIMES))
    regret_graph = np.zeros((len(agent), EPISODE_TIMES))

    # トレーニング開始
    print("トレーニング開始")
    for simu in range(SIMULATION_TIMES):
        for n_agent in range(len(agent)):
            agent[n_agent].init_params()  # エージェントのパラメータを初期化
            # 信頼値初期化
            # tau = np.zeros((task.get_num_state()+1, task.get_num_action()))
            regret = 0
            agent[n_agent].init_tau()
            for epi in range(EPISODE_TIMES):
                current_state = task.get_start()  # 現在地をスタート地点に初期化
                probability = 1
                while True:
                    # 行動選択
                    action = agent[n_agent].serect_action(current_state)
                    # 報酬を観測
                    reward = task.get_reward(current_state, action)
                    reward_graph[n_agent, epi] += reward
                    # 次状態を観測
                    next_state = task.get_next_state(current_state, action)
                    next_action = agent[n_agent].serect_action(next_state)
                    # Q価の更新
                    agent[n_agent].update(current_state, action, reward, next_state, next_action)
                    # 選んだ行動の当たる確率
                    probability *= task.get_probability(current_state, action)
                    # 次状態が終端状態であれば終了
                    if next_state == task.get_goal_state():
                        regret += max - probability
                        regret_graph[n_agent, epi] += regret
                        break
                    # print("action: ", action, "next: ", next_state)
                    # print(probability)
                    current_state = next_state
            # agent[n_agent].get_tau()

    print("Q値の表示:PS")
    agent[0].print_value()
    print("Q値の表示:RS")
    agent[1].print_value()

    # Q_learning
    print("グラフ表示")
    plt.plot(reward_graph[0] / SIMULATION_TIMES, label="PS")
    plt.plot(reward_graph[1] / SIMULATION_TIMES, label="RS")
    plt.legend()  # 凡例を付ける
    plt.title("Q_learning")  # グラフタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルを付ける
    plt.ylabel("sum reward")  # y軸のラベルを付ける
    plt.show()  # グラフを表示


    """# regret 確率max-actionの総和
    # max = 0.4 * 0.9
    print("グラフ表示")
    plt.plot(regret_graph[0]/SIMULATION_TIMES, label="PS")
    plt.plot(regret_graph[1]/SIMULATION_TIMES, label="RS")
    plt.legend()  # 凡例を付ける
    plt.title("Q_learning")  # グラフタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルを付ける
    plt.ylabel("regret")  # y軸のラベルを付ける
    plt.show()  # グラフを表示"""

    """# Sarsa
    print("グラフ表示")
    plt.plot(reward_graph[0] / SIMULATION_TIMES, label="PS")
    plt.plot(reward_graph[1] / SIMULATION_TIMES, label="RS")
    plt.legend()  # 凡例を付ける
    plt.title("Sarsa")  # グラフタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルを付ける
    plt.ylabel("sum reward")  # y軸のラベルを付ける
    plt.show()  # グラフを表示"""

    """print("グラフ表示")
    plt.plot(regret_graph[0]/SIMULATION_TIMES, label="PS")
    plt.plot(regret_graph[1]/SIMULATION_TIMES, label="RS")
    plt.legend()  # 凡例を付ける
    plt.title("Sarsa")  # グラフタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルを付ける
    plt.ylabel("regret")  # y軸のラベルを付ける
    plt.show()  # グラフを表示"""

main()
