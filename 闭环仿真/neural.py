"""可审计的固定受体编码与经原反向赫布译码的四肌肉运动接口。

前向来源、激励/抑制权重、阈值、读出权重在出生时设定。
只有 MotorBank.reverse.矩阵 由原反向赫布网络的一次配对学习产生。
没有语义分类器；所有模式由成对受体、固定求和和阈值计算得到。
"""
from __future__ import annotations

import numpy as np
import 运动输出区 as legacy_motor


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _thermometer(values: np.ndarray, channels: int, levels: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (channels,) or not np.isfinite(values).all():
        raise ValueError(f"expected {channels} finite channel values")
    quantized = np.rint(np.clip(values, 0.0, 1.0) * levels).astype(int)
    positive = np.arange(levels)[None, :] < quantized[:, None]
    signal = np.empty((channels, levels, 2), dtype=float)
    signal[:, :, 0] = positive
    signal[:, :, 1] = ~positive
    return signal.ravel()


def _network_arrays(network, prefix: str) -> dict[str, np.ndarray]:
    result = {}
    for layer in range(network.层数 - 1):
        result[f"{prefix}.source.{layer}"] = network.来源[layer]
        result[f"{prefix}.weight.{layer}"] = network.权重[layer]
        result[f"{prefix}.threshold.{layer}"] = network.阈值[layer]
    return result


class PairedEncoder:
    """0~1 通道 -> 温度计竞争 -> 相邻边界检测，每通道恰一输出。

    一通道占 levels*2 个位置。零值输出在首 odd；等级1~levels
    分别输出在对应 even。边界检测使用固定 pos[j]-pos[j+1] 突触，
    阈值为0.5，末级直接读末个正受体，其余 odd 禁用。
    """

    def __init__(self, channels: int, levels: int = 10):
        self.channels = _count(channels, "channels")
        self.levels = _count(levels, "levels")
        self.n = self.channels * self.levels * 2
        self.pool = legacy_motor.特征神经元池(self.channels * self.levels)
        self.network = legacy_motor.神经网络(
            self.pool, 层数=2, 连接半径=3, 权重范围=0.0, 阈值初值=2.0
        )
        source = self.network.来源[0]
        weights = self.network.权重[0]
        thresholds = self.network.阈值[0]
        weights[:] = 0.0
        self.readout = np.zeros((self.channels, self.n), dtype=float)

        def connect(target: int, origin: int, weight: float):
            candidates = np.flatnonzero(source[target] == origin)
            if candidates.size == 0:
                raise RuntimeError("fixed local receptive field misses a boundary")
            # 小维度环可能让同一来源出现两次，只布一条线。
            weights[target, candidates[0]] = weight

        for channel in range(self.channels):
            base = channel * self.levels * 2
            thresholds[base + 1] = 0.5
            connect(base + 1, base + 1, 1.0)
            for level in range(self.levels):
                target = base + 2 * level
                thresholds[target] = 0.5
                connect(target, target, 1.0)
                if level + 1 < self.levels:
                    connect(target, target + 2, -1.0)
                self.readout[channel, target] = (level + 1) / self.levels

    def encode(self, values) -> np.ndarray:
        signal = _thermometer(values, self.channels, self.levels)
        return self.network.前向传播(signal).astype(bool, copy=True)

    def decode(self, output) -> np.ndarray:
        output = np.asarray(output, dtype=float)
        if output.shape != (self.n,) or not np.isfinite(output).all():
            raise ValueError(f"expected {self.n} finite neural activations")
        # 固定受体读出；正常encode返回每通道一个激活，值自然在0~1。
        return np.clip(self.readout @ output, 0.0, 1.0)

    def fixed_arrays(self) -> dict[str, np.ndarray]:
        result = _network_arrays(self.network, "paired")
        result["paired.readout"] = self.readout
        return result


class MotorBank:
    """固定组合检测产生命令，原反向赫布经一次配对把命令译成肌肉。

    肌肉顺序必须与world一致：[left_fwd,left_back,right_fwd,right_back]。
    行为索引：0静息、1前进、2后退、3左转、4右转。
    模板仅用于出生运动配对；执行decode始终读取可塑反向矩阵。
    """

    muscles = 4
    action_count = 5

    def __init__(self, levels: int = 10):
        self.levels = _count(levels, "levels")
        self.n = self.muscles * self.levels * 2
        if self.n < self.action_count:
            raise ValueError("not enough neurons for motor commands")
        self.templates = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.6, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.3, 0.3, 0.0],
            [0.3, 0.0, 0.0, 0.3],
        ], dtype=float)
        self.pool = legacy_motor.特征神经元池(self.muscles * self.levels)
        self.network = legacy_motor.神经网络(
            self.pool, 层数=2, 连接半径=0, 权重范围=0.0, 阈值初值=1.1
        )
        self.network.来源[0] = np.tile(np.arange(self.n), (self.n, 1))
        self.network.权重[0] = np.zeros((self.n, self.n), dtype=float)
        for action, template in enumerate(self.templates):
            # 完整组合AND：所有应亮的成对受体都亮时，归一化和为1。
            self.network.权重[0][action] = (
                _thermometer(template, self.muscles, self.levels) / self.pool.对数
            )
            self.network.阈值[0][action] = 1.0 - 1e-8
        self.reverse = legacy_motor.反向赫布网络(
            legacy_motor.特征神经元池(self.muscles * self.levels),
            本底范围=0.0,
            一次学会=True,
        )
        # 出生静息偏置：没有命令/连接时反受体获胜。有效学习票为1，
        # 明显大于0.01，所以不会改变已学动作。避免原版平票全正发力。
        self.reverse.本底[1::2] = 0.01
        for action, template in enumerate(self.templates):
            command = self.encode(template)
            if np.flatnonzero(command).tolist() != [action]:
                raise ValueError("motor levels collapse distinct action templates")
            self.reverse.学习一帧(self.pool.激活.copy(), command)

    def encode(self, forces) -> np.ndarray:
        signal = _thermometer(forces, self.muscles, self.levels)
        return self.network.前向传播(signal).astype(bool, copy=True)

    def command(self, action_index: int) -> np.ndarray:
        if isinstance(action_index, bool) or int(action_index) != action_index:
            raise ValueError("action index must be an integer")
        action_index = int(action_index)
        if not 0 <= action_index < self.action_count:
            raise ValueError("action index outside motor bank")
        command = np.zeros(self.n, dtype=bool)
        command[action_index] = True
        return command

    def decode_command(self, command) -> np.ndarray:
        command = np.asarray(command, dtype=bool)
        if command.shape != (self.n,):
            raise ValueError(f"expected a motor command with {self.n} neurons")
        receptor = self.reverse.前向(command)
        return receptor[0::2].reshape(self.muscles, self.levels).sum(axis=1) / self.levels

    def decode(self, action_index: int) -> np.ndarray:
        return self.decode_command(self.command(action_index))

    def fixed_arrays(self) -> dict[str, np.ndarray]:
        result = _network_arrays(self.network, "motor")
        result["motor.templates"] = self.templates
        result["motor.resting_bias"] = self.reverse.本底
        return result


def _self_check():
    """输入边界、运动精确译码、断线因果性和出生权重边界自检。"""
    import hashlib

    def checksum(arrays):
        digest = hashlib.sha256()
        for name, array in sorted(arrays.items()):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    rng = np.random.default_rng(20260906)
    checks = 0
    for channels, levels in ((1, 1), (1, 10), (4, 10), (21, 8)):
        encoder = PairedEncoder(channels, levels)
        before = checksum(encoder.fixed_arrays())
        samples = [np.full(channels, k / levels) for k in range(levels + 1)]
        samples += list(rng.uniform(-0.5, 1.5, (20, channels)))
        for values in samples:
            output = encoder.encode(values)
            assert output.reshape(channels, levels * 2).sum(axis=1).tolist() == [1] * channels
            expected = np.rint(np.clip(values, 0, 1) * levels) / levels
            np.testing.assert_allclose(encoder.decode(output), expected, atol=1e-12)
            assert np.all(encoder.pool.激活[0::2] ^ encoder.pool.激活[1::2])
            checks += 1
        assert checksum(encoder.fixed_arrays()) == before
    motor = MotorBank()
    before = checksum(motor.fixed_arrays())
    for action, forces in enumerate(motor.templates):
        assert np.flatnonzero(motor.encode(forces)).tolist() == [action]
        np.testing.assert_allclose(motor.decode(action), forces, atol=1e-12)
    np.testing.assert_array_equal(motor.decode_command(np.zeros(motor.n, dtype=bool)), np.zeros(4))
    learned_connections = motor.reverse.条数()
    motor.reverse.矩阵[:] = False
    for action in range(motor.action_count):
        np.testing.assert_array_equal(motor.decode(action), np.zeros(4))
    assert checksum(motor.fixed_arrays()) == before
    print(f"PairedEncoder: {checks} patterns, exclusive pairs and exact quantization PASS")
    print(f"MotorBank: n={motor.n}, 5/5 exact commands, {learned_connections} learned reverse edges PASS")
    print("Zero command and reverse-connection ablation -> four relaxed muscles PASS")
    print("All birth-fixed sources, weights, thresholds, readout and bias unchanged PASS")


if __name__ == "__main__":
    _self_check()
