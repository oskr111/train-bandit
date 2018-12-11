"""
python3
バンディット問題を強化学習するプログラム

スロットマシンは２つ存在していて、それぞれの確率は[0, 1]
当たると報酬が１もらえる

方策はgreedyとε-greedyとε減衰の３つ(このコードにはgreedyしか実装されていない)

PSとRSの実装

PS（素朴満足化ポリシー）

行動決定の時に基準値を
・超えている
    ランダムorE-greedy
・超えていない
    greedyな行動を選択

RS（満足化価値関数）
PSに信頼度が入る

基準値について
取らせたいもの[0.4, 0.6]だと0.5を基準値に設定すれば
0.6をずっと取れるよね

"""

import numpy as np
import matplotlib.pyplot as plt
import random


# バンディットタスク
class Bandit(object):
    def __init__(self):
        # バンディットの設定
        self.probability = np.asarray([[0.4, 0.6]])
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
        return current_state + 1


class Addbandit(Bandit):
    def __init__(self):
        # バンディットの設定
        self.probability = np.asarray([[0.4, 0.6]])
        # スタート地点
        self.start = 0
        # ゴール地点
        self.goal = len(self.probability)


class Treebandit(Bandit):
    def __init__(self):
        # バンディットの設定
        self.probability = np.asarray([[0.4, 0.5, 0.6, 0.7]])
        # スタート地点
        self.start = 0
        # ゴール地点
        self.goal = len(self.probability)




# Q学習のクラス
class Q_learning(object):
    # 学習率、割引率、状態数、行動数を定義する、def_init_で初期化
    def __init__(self, learning_rate=0.1, discount_rate=0.9, num_state=None, num_action=None):
        self.learning_rate = learning_rate  # 学習率
        self.discount_rate = discount_rate  # 割引率
        self.num_state = num_state  # 状態数
        self.num_action = num_action  # 行動数
        # Qテーブルを初期化
        self.Q = np.zeros((self.num_state+1, self.num_action))

    # Q値の更新
    # 現状態、選択した行動、得た報酬、次状態を受け取って更新する
    def update_Q(self, current_state, current_action, reward, next_state):
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




# 方策クラス
class Greedy(object):  # greedy方策
    # 行動価値を受け取って行動番号を返す
    def serect_action(self, value, current_state):
        return np.argmax(value[current_state])


class E_greedy(Greedy): #egreedy方策
    def __init__(self,eps):
        self.eps = eps

    def serect_action(self,value,current_state):

        e = random.random()

        if e <= self.eps:
            return np.random.choice(range(len(value[current_state])))
        elif e > self.eps:
            return np.argmax(value[current_state])


class E_decay(E_greedy):  #E減衰
    def __init__(self,eps):
        super().__init__(eps)
        self.eps = eps

    def serect_action(self,value,current_state):
        self.eps = self.eps - 1/1000
        e = random.random()

        if e <= self.eps:
            return np.random.choice(range(len(value[current_state])))
        elif e > self.eps:
            return np.argmax(value[current_state])


# PS
class PS(object):
    def __init__(self, r_state, eps):
        self.r_state = r_state      # 基準値
        self.eps = eps


    # 行動選択　PSを入れる
    def serect_action(self,value,current_state):

        q_value = max(value[current_state])

        if self.r_state < q_value:
             e = random.random()

             if e <= self.eps:
                 return np.random.choice(range(len(value[current_state])))
             elif e > self.eps:
                 return np.argmax(value[current_state])
        else:
            return np.argmax(value[current_state])


# RS
class RS(PS):
    def __init__(self, r_state):
        self.r_state = r_state      # 基準値
        self.tau_state = [0,0,0,0]
        tau = [0,1,2,3]

    # 行動選択　RSを入れる
    # value=Q であり　Qの中身はQ[current_state, current_action]
    def serect_action(self,value,current_state):


        tau_action = np.where(value[1] == np.max(value[1]))
        print("value[1]: {}", value[1])
        print(format(tau_action))

        if tau_action == 0:
            self.tau_state[0] = self.tau_state[0]+1
        elif tau_action == 1:
            self.tau_state[1] = self.tau_state[1]+1
        elif tau_action == 2:
            self.tau_state[2] = self.tau_state[2]+1
        else:
            self.tau_state[3] = self.tau_state[3]+1
       # もっともtauの中で大きいものの場所
        #tau_before = np.where(self.tau_state == np.max(self.tau_state))
        #tau0 = random.choice(tau_before[0])

        tau = self.tau_state

        RS=[0,0,0,0]
        # RS値そのもの

        i = 0
        for i in range(4):
            RS[i] = tau[i]*(value[current_state] - self.r_state)

        idx = np.where(RS == np.max(RS))

        return random.choice(idx[0])

    # serect_actionにtauを作る np.where





# エージェントクラス
class Agent():
    def __init__(self, value_func="Q_learning", policy="RS", learning_rate=0.1, discount_rate=0.9, n_state=None, n_action=None):
        # 価値更新方法の選択
        if value_func == "Q_learning":
            self.value_func = Q_learning(num_state=n_state, num_action=n_action)

        # 方策の選択
        if policy == "greedy":
            self.policy = Greedy()

        elif policy == "E_greedy":
            self.policy = E_greedy(0.1)

        elif policy == "E_decay":
            self.policy = E_decay(1.0)

        elif policy == "PS":
            self.policy = PS(0.5, 0.1)

        elif policy == "RS":
            self.policy = RS(0.5)



    # パラメータ更新(基本呼び出し)
    def update(self, current_state, current_action, reward, next_state):
        self.value_func.update_Q(current_state, current_action, reward, next_state)

    # 行動選択(基本呼び出し)
    def serect_action(self, current_state):
        return self.policy.serect_action(self.value_func.get_Q(), current_state)

    # 行動価値の表示
    def print_value(self):
        print(self.value_func.get_Q())

    # 所持パラメータの初期化
    def init_params(self):
        self.value_func.init_params()


# メイン関数
def main():
    # ハイパーパラメータ等の設定
    task = Treebandit()  # タスク定義

    SIMULATION_TIMES = 1  # シミュレーション回数
    EPISODE_TIMES = 100  # エピソード回数
    xnum_action = 0
    ynum_action = 0
    xsum = 0
    ysum = 0


    agent = Agent(n_state=task.get_num_state(), n_action=task.get_num_action())  # エージェントの設定

    sumreward_graph = np.zeros(EPISODE_TIMES)  # グラフ記述用の報酬記録

    # トレーニング開始
    print("トレーニング開始")
    for simu in range(SIMULATION_TIMES):
        agent.init_params()  # エージェントのパラメータを初期化
        for epi in range(EPISODE_TIMES):
            current_state = task.get_start()  # 現在地をスタート地点に初期化

            while True:
                # 行動選択
                action = agent.serect_action(current_state)
                # 報酬を観測
                reward = task.get_reward(current_state, action)
                sumreward_graph[epi] += reward
                # 次状態を観測
                next_state = task.get_next_state(current_state, action)
                # Q価の更新
                agent.update(current_state, action, reward, next_state)
                # 次状態が終端状態であれば終了


                if action == 0:
                    xnum_action = xnum_action + 1
                    if reward == 1:
                        xsum = xsum + 1

                if action == 1:
                    ynum_action = ynum_action + 1
                    if reward == 1:
                        ysum = ysum + 1



                if next_state == task.get_goal_state():
                    break

    print("Q値の表示")
    agent.print_value()

    #x = xsum/xnum_action
    #y = ysum/ynum_action

    print("報酬確率")
    #print("[{}, {}]" .format(x,y))

    print("グラフ表示")
    plt.plot(sumreward_graph / SIMULATION_TIMES, label="E_greedy")  # グラフ書き込み
    plt.legend()  # 凡例を付ける
    plt.title("reward")  # グラフタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルを付ける
    plt.ylabel("sum reward")  # y軸のラベルを付ける
    plt.show()  # グラフを表示


main()
