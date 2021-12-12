__author__ = "GitHub@laorange"
__license__ = "AGPL-3.0 License"

import pandas as pd
import plotly
import plotly.express as px
import random
from typing import Union, Dict, Callable
import numpy as np
from tqdm import tqdm


class Util:
    @staticmethod
    def multiple_decorator(func_outputting_bool, *parameters):
        for parameter in parameters:
            if isinstance(parameter, list) or isinstance(parameter, tuple):
                if func_outputting_bool(*parameter):
                    continue
            elif func_outputting_bool(parameter):
                continue
            return False
        return True

    @staticmethod
    def is_int(num):
        if isinstance(num, float) or isinstance(num, int):
            if int(num) == num:
                return True
        return False

    @staticmethod
    def get_decimal_precision(_num: float):
        return str(_num)[::-1].find(".")


class Config(object):
    # region TODO: ↓在此区域定义参数
    h1: float = 0.02  # x方向的网格宽度
    h2: float = 0.02  # y方向的网格高度

    # 下方最大值最小值必须留有余量，否则会报错！
    omega_x_min: float = -3  # Ω区域在x方向的最小值
    omega_x_max: float = 3  # Ω区域在x方向的最大值
    omega_y_min: float = -3  # Ω区域在y方向的最小值
    omega_y_max: float = 3  # Ω区域在y方向的最大值

    monte_carlo_method_num = 200  # 蒙特卡罗方法的随机次数

    @staticmethod
    def in_omega(x: Union[int, float], y: Union[int, float]) -> bool:
        """该函数返回True/False；True表明该点在Ω中，False表明不在Ω中"""
        return x ** 2 + y ** 2 - 1 <= 0

    @staticmethod
    def a(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return np.sin(x) + np.cos(y)

    @staticmethod
    def b(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return np.sin(2 * x) + np.cos(2 * y)

    @staticmethod
    def c(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return np.sin(x) * np.cos(y)

    @staticmethod
    def d(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return np.sin(2 * x) * np.cos(2 * y)

    @staticmethod
    def f(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return np.sin(x) + np.cos(y)

    @staticmethod
    def g(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return x ** 2 + y ** 2

    # endregion

    @staticmethod
    def not_in_omega(x: Union[int, float], y: Union[int, float]) -> bool:
        return not Config.in_omega(x, y)

    x_amount = (omega_x_max - omega_x_min) / h1
    y_amount = (omega_y_max - omega_y_min) / h1
    assert Util.multiple_decorator(Util.is_int, x_amount, y_amount)


class Point(object):
    instance_list = []

    def __init__(self, x: Union[int, float], y: Union[int, float], i: int, j: int):
        self.x, self.y, self.i, self.j = x, y, i, j
        self.id_num = len(Point.instance_list)
        Point.instance_list.append(self)
        self.value = None

    def set_value(self, value: Union[int, float]):
        self.value = value

    @staticmethod
    def plot_finally():
        x_list = [point.x for point in Point.instance_list]
        y_list = [point.y for point in Point.instance_list]
        value_list = [point.value for point in Point.instance_list]
        df = pd.DataFrame({"x": x_list, "y": y_list, "value": value_list,
                           "color": np.sqrt(np.abs(np.array(x_list)) ** 2 + np.abs(np.array(y_list)) ** 2)})
        figure_3d = px.scatter_3d(df, x="x", y="y", z="value", color="color")
        plotly.offline.plot(figure_3d, filename=f"output_3d.html")


class PointWithBoundaryCondition(Point):
    def __init__(self, x: Union[int, float], y: Union[int, float], i: int, j: int):
        super().__init__(x, y, i, j)
        self.value = Config.g(x, y)

    def set_value(self, value: Union[int, float]):
        if abs(value - self.value) > 1e-6:
            print(f"({self.x},{self.y})与边界条件冲突！边界条件：{self.value} | 计算值：{value}")


class Table(object):
    @staticmethod
    def monte_carlo_method(x: Union[int, float], y: Union[int, float]) -> bool:
        """使用Monte Carlo方法判断某区域是否有50%以上在Ω范围内。x, y是该区域左下角的坐标"""
        record_num = 0
        for _ in range(Config.monte_carlo_method_num):
            _x = x + random.random() * Config.h1
            _y = y + random.random() * Config.h2
            if Config.in_omega(_x, _y):
                record_num += 1
        return (record_num / Config.monte_carlo_method_num) >= 0.5

    @staticmethod
    def whether_fill_a_block(i: Union[int, float], j: Union[int, float]) -> bool:
        four_point = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
        if Util.multiple_decorator(Config.in_omega, *four_point):
            return True
        elif Util.multiple_decorator(Config.not_in_omega, *four_point):
            return False
        return Table.monte_carlo_method(i, j)

    def __init__(self, x_list: np.ndarray, y_list: np.ndarray, block_marks: np.ndarray):
        self.x_list: np.ndarray = x_list
        self.y_list: np.ndarray = y_list
        self.block_marks = block_marks

        self.point_dict: Dict[str, Point] = {}
        self.parse_block_marks()
        dimension = len(self.point_dict)

        # A · U = F => U = A^-1 · F
        self.matrix_a = np.zeros((dimension, dimension))
        self.matrix_f = np.zeros((dimension, 1))
        self.matrix_u = None

    def plot_block_marks(self):
        df = pd.DataFrame({"x": self.x_list.ravel(), "y": self.y_list.ravel(),
                           "z": self.block_marks.ravel()})
        fig = px.scatter(df, x="x", y="y", color="z")
        fig.show()

    def parse_block_marks(self):
        for i, _ in enumerate(self.block_marks):
            for j, __ in enumerate(_):
                x = self.x_list[i][j]
                y = self.y_list[i][j]
                count = 0
                for point in [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1),
                              (i - 1, j + 1), (i + 1, j + 1), (i - 1, j - 1), (i + 1, j - 1)]:
                    if point[0] < 0 or point[1] < 0 or point[0] >= Config.x_amount or point[1] >= Config.y_amount:
                        continue
                    elif self.block_marks[point[0]][point[1]] == 1:
                        count += 1
                if 0 < count < 8 and self.block_marks[i][j] == 1:
                    self.point_dict[f"{i},{j}"] = PointWithBoundaryCondition(x, y, i, j)
                elif count == 8:
                    self.point_dict[f"{i},{j}"] = Point(x, y, i, j)

    def solve(self):
        for coordinate, point in tqdm(self.point_dict.items(), desc="正在更新A矩阵..."):
            self.update(point)
        print("正在求A矩阵的逆...")
        self.matrix_u = np.dot(np.linalg.inv(self.matrix_a), self.matrix_f)
        print("求解完成！即将开始画图...")
        for _index, u in enumerate(self.matrix_u.ravel()):
            Point.instance_list[_index].set_value(u)

    def update(self, point: Point):
        id_num = point.id_num
        if isinstance(point, PointWithBoundaryCondition):
            self.matrix_f[id_num][0] = Config.g(point.x, point.y)
            self.matrix_a[id_num][id_num] = 1
        else:
            self.matrix_f[id_num][0] = Config.f(point.x, point.y)

            self.matrix_a[id_num][self.point_dict[f"{point.i - 1},{point.j - 1}"].id_num] = self.get_different_schema(0)(point)
            self.matrix_a[id_num][self.point_dict[f"{point.i - 1},{point.j}"].id_num] = self.get_different_schema(1)(point)
            self.matrix_a[id_num][self.point_dict[f"{point.i - 1},{point.j + 1}"].id_num] = self.get_different_schema(2)(point)
            self.matrix_a[id_num][self.point_dict[f"{point.i},{point.j - 1}"].id_num] = self.get_different_schema(3)(point)
            self.matrix_a[id_num][self.point_dict[f"{point.i},{point.j}"].id_num] = self.get_different_schema(4)(point)
            self.matrix_a[id_num][self.point_dict[f"{point.i},{point.j + 1}"].id_num] = self.get_different_schema(5)(point)
            self.matrix_a[id_num][self.point_dict[f"{point.i + 1},{point.j - 1}"].id_num] = self.get_different_schema(6)(point)
            self.matrix_a[id_num][self.point_dict[f"{point.i + 1},{point.j}"].id_num] = self.get_different_schema(7)(point)
            self.matrix_a[id_num][self.point_dict[f"{point.i + 1},{point.j + 1}"].id_num] = self.get_different_schema(8)(point)

    @staticmethod
    def get_different_schema(mark_num) -> Callable:
        schema = [
            lambda point: 2 * Config.b(point.x, point.y) / (Config.h1 * Config.h2),
            lambda point: Config.a(point.x, point.y) / Config.h1 ** 2 - Config.d(point.x, point.y) / Config.h1,
            lambda point: 2 * Config.b(point.x, point.y) / (Config.h1 * Config.h2),
            lambda point: Config.c(point.x, point.y) / (Config.h2 ** 2),
            lambda point: -(2 * Config.a(point.x, point.y) / Config.h1 ** 2 + 2 * Config.c(point.x, point.y) / Config.h2 ** 2),
            lambda point: Config.c(point.x, point.y) / (Config.h2 ** 2),
            lambda point: -2 * Config.b(point.x, point.y) / (Config.h1 * Config.h2),
            lambda point: Config.a(point.x, point.y) / Config.h1 ** 2 + Config.d(point.x, point.y) / Config.h1,
            lambda point: 2 * Config.b(point.x, point.y) / (Config.h1 * Config.h2),
        ]
        return schema[mark_num]


class Handler(object):
    def __init__(self):
        x_list, y_list = np.mgrid[Config.omega_x_min:Config.omega_x_max:Config.h1, Config.omega_y_min:Config.omega_y_max:Config.h2]
        block_marks = np.zeros(x_list.shape)
        for index_i, i in enumerate(tqdm(x_list, desc="正在将Ω区域锯齿化...")):
            for index_j, j in enumerate(y_list[0]):
                block_marks[index_i][index_j] = int(Table.whether_fill_a_block(i[0], j))

        precision_x = Util.get_decimal_precision(Config.h1)
        precision_y = Util.get_decimal_precision(Config.h2)
        self.table = Table(np.round(x_list, precision_x), np.round(y_list, precision_y),
                           np.round(block_marks, 2 * (precision_x + precision_y)))


if __name__ == '__main__':
    handler = Handler()
    handler.table.solve()
    Point.plot_finally()
    print("完成！")
