from ca import CellularAutomata
from random import randint

# 在此处修改代码，调用 test 得到的返回值将作为运行结果返回主程序。

def rule() -> int:
    return randint(2 ** (2 ** 20), 2 ** (2 ** 21))

def test():
    meta = CellularAutomata(
        calculate_module="numpy",
        rule=rule(),
        time=32,
        size=[30, 30, 30],
        log_rate=16,
        is_saving=True,
        is_save=True,
        start_pos=[10, 10, 10]
    )
    result = meta.run("mark")
    return result
