import numpy as np
from 海马体时间区 import 时钟
from 权重连接管理 import 连接表
from 视觉前处理 import 网


class 记忆区:
    """记忆区：特征神经元 + 时序赫布权重；时间由共享时钟（海马体时间区）驱动"""

    def __init__(self, 神经元数量):
        self.总数 = 神经元数量
        self.激活 = np.zeros(神经元数量, dtype=bool)   # 特征神经元激活状态
        self.特征到时间 = 连接表()   # 特征编号 → {时间编号: 权重}（带定期遗忘）
        self.时间到特征 = 连接表()   # 时间编号 → {特征编号: 权重}（带定期遗忘）

    def 学习(self, 旧时间, 新时间, 新特征):
        """一个内部时间步（0.05 秒）的时序赫布学习。
        旧时间/新时间由共享时钟给出；新特征 = 这一帧记忆区的特征激活状态。
        """
        # 规则：上一帧的特征 → 下一格时间神经元
        for f in np.nonzero(self.激活)[0]:
            self.特征到时间.学习(f, 新时间)

        # 规则：上一格时间神经元 → 下一帧的特征
        for f in np.nonzero(新特征)[0]:
            self.时间到特征.学习(旧时间, f)

        self.激活[:] = 新特征
        self.特征到时间.维护()   # 连接达到警戒线后定期整体衰减，防无限增长
        self.时间到特征.维护()

    def 回忆(self, 特征模式, 步数=10, 阈值=1):
        """给一组特征，找到关联时间，沿共享时间环连续播放"""
        得分 = np.zeros(时钟.时间神经元数量)
        for f in np.nonzero(特征模式)[0]:
            for t, w in self.特征到时间.查(f).items():
                得分[t] += w

        if 得分.max() == 0:
            return []   # 这些特征没有学过任何时间关联

        当前时间 = int(np.argmax(得分))
        播放 = []
        for _ in range(步数):
            特征 = np.zeros(self.总数, dtype=bool)
            for f, w in self.时间到特征.查(当前时间).items():
                if w >= 阈值:
                    特征[f] = True
            播放.append((当前时间, 特征))
            当前时间 = (当前时间 + 1) % 时钟.时间神经元数量
        return 播放


if __name__ == "__main__":
    print("时间神经元数量:", 时钟.时间神经元数量)
    print("一帧秒数:", 时钟.每步秒数 * 时钟.每帧步数)
    print("绕一圈小时:", 时钟.时间神经元数量 * 时钟.每步秒数 / 3600)

    记忆 = 记忆区(网.输出层数量)

    # 造 4 帧记忆：A → B → C → D（每帧 2 个特征神经元，方便观察）
    模式A = np.zeros(记忆.总数, dtype=bool); 模式A[[10, 20]] = True
    模式B = np.zeros(记忆.总数, dtype=bool); 模式B[[30, 40]] = True
    模式C = np.zeros(记忆.总数, dtype=bool); 模式C[[50, 60]] = True
    模式D = np.zeros(记忆.总数, dtype=bool); 模式D[[70, 80]] = True

    # 主循环驱动方式：时钟走一格，记忆区学一格（一帧 = 每帧步数 格）
    for 模式 in [模式A, 模式B, 模式C, 模式D]:
        for _ in range(时钟.每帧步数):
            旧时间, 新时间 = 时钟.走一步()
            记忆.学习(旧时间, 新时间, 模式)
    print("学习完成：A → B → C → D")

    播放 = 记忆.回忆(模式A, 步数=8)
    for 时间, 特征 in 播放:
        print("时间", 时间, "→ 特征:", np.nonzero(特征)[0])
