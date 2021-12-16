__author__ = "GitHub@laorange"
__license__ = "AGPL-3.0 License"

import random
import datetime
from typing import Union, Dict, Tuple, List

import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly
import plotly.graph_objects as go


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

    @staticmethod
    def print(*args):
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S : "), *args)

    @staticmethod
    def delete_all_0_col_and_raw_for_2d_array(zs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx0: List[int] = list(np.argwhere(np.all(zs[..., :] == 0, axis=1)).ravel())
        xs = np.delete(Table.xs, idx0, axis=0)
        zs = np.delete(zs, idx0, axis=0)
        idx1: List[int] = list(np.argwhere(np.all(zs[..., :] == 0, axis=0)).ravel())
        zs = np.delete(zs, idx1, axis=1)
        ys = np.delete(Table.xs, idx1, axis=0)
        return xs, ys, zs


class Config(object):
    # region TODO: ↓在此区域定义参数
    h1: float = 0.02  # x方向的网格宽度
    h2: float = 0.02  # y方向的网格高度

    omega_x_min: float = -4  # Ω区域在x方向的最小值
    omega_x_max: float = 4  # Ω区域在x方向的最大值
    omega_y_min: float = -4  # Ω区域在y方向的最小值
    omega_y_max: float = 4  # Ω区域在y方向的最大值

    error_tolerance: float = 1e-5  # 边界条件校验时允许的误差上限

    monte_carlo_method_num = 200  # 蒙特卡罗方法的随机次数

    @staticmethod
    def in_omega(x: Union[int, float], y: Union[int, float]) -> bool:
        """该函数返回True/False；True表明该点在Ω中，False表明不在Ω中"""
        return x ** 2 + y ** 2 <= 1 and (abs(x * y) < 0.25)

    @staticmethod
    def a(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return (1 + (np.exp(-(x * y)))) ** -1

    @staticmethod
    def b(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return np.sin(2 * x) + np.cos(3 * y)

    @staticmethod
    def c(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return x ** 2 + y ** 2

    @staticmethod
    def d(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
        return (1 + (np.exp(-(x + y)))) ** -1

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

    @staticmethod
    def round_x(num: Union[np.array, int, float]):
        precision_x = Util.get_decimal_precision(Config.h1)
        return np.round(num, precision_x)

    @staticmethod
    def round_y(num: Union[np.array, int, float]):
        precision_y = Util.get_decimal_precision(Config.h2)
        return np.round(num, precision_y)

    x_amount = (omega_x_max - omega_x_min) / h1
    y_amount = (omega_y_max - omega_y_min) / h2


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
        df = pd.DataFrame({"x": x_list, "y": y_list, "value": value_list})
        df.to_csv("result.csv")
        Util.print("结果已导出到当前目录下的csv文件中，即将开始画图...")

        table_array = np.zeros(Table.x_list.shape)
        for i, _ in enumerate(table_array):
            for j, __ in enumerate(_):
                table_array[i][j] = np.NAN
        for point in Point.instance_list:
            table_array[point.i][point.j] = point.value
        xs, ys, table_array = Util.delete_all_0_col_and_raw_for_2d_array(table_array)
        fig = go.Figure(data=[go.Surface(z=table_array, x=xs, y=ys)])
        fig.update_layout(title='result', autosize=True)
        fig.update_traces(contours_z=dict(show=False, usecolormap=True,
                                          highlightcolor="limegreen", project_z=True))
        plotly.offline.plot(fig, filename=f"output_3d_surface.html")


class PointWithBoundaryCondition(Point):
    def __init__(self, x: Union[int, float], y: Union[int, float], i: int, j: int):
        super().__init__(x, y, i, j)
        self.value = Config.g(x, y)

    def set_value(self, value: Union[int, float]):
        if abs(value - self.value) > Config.error_tolerance:
            Util.print(f"({self.x},{self.y})与边界条件冲突！边界条件：{self.value} | 计算值：{value}")


class Table(object):
    xs = np.linspace(Config.omega_x_min, Config.omega_x_max, int(Config.x_amount))
    ys = np.linspace(Config.omega_y_min, Config.omega_y_max, int(Config.y_amount))
    x_list, y_list = np.mgrid[
                     Config.omega_x_min:Config.omega_x_max:Config.h1,
                     Config.omega_y_min:Config.omega_y_max:Config.h2]

    def __init__(self, block_marks: np.ndarray):
        self.block_marks = block_marks

        self.point_dict: Dict[str, Point] = {}
        self.parse_block_marks()
        dimension = len(self.point_dict)

        # A · U = F => U = A^-1 · F
        self.matrix_a = np.zeros((dimension, dimension))
        self.matrix_f = np.zeros((dimension, 1))
        self.matrix_u = None

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

    def plot_block_marks(self):
        xs, ys, zs = Util.delete_all_0_col_and_raw_for_2d_array(self.block_marks)
        fig = go.Figure(data=[go.Heatmap(x=xs, y=ys, z=zs)])
        fig.update_layout(title='table', autosize=True)
        plotly.offline.plot(fig, filename=f"output_table.html")

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

    def update(self, point: Point):
        id_num = point.id_num
        if isinstance(point, PointWithBoundaryCondition):
            self.matrix_f[id_num][0] = Config.g(point.x, point.y)
            self.matrix_a[id_num][id_num] = 1
        else:
            self.matrix_f[id_num][0] = Config.f(point.x, point.y)

            schema_list = [
                [lambda _p: 2 * Config.b(_p.x, _p.y) / (Config.h1 * Config.h2),
                 lambda _p: Config.a(_p.x, _p.y) / Config.h1 ** 2 - Config.d(_p.x, _p.y) / Config.h1,
                 lambda _p: 2 * Config.b(_p.x, _p.y) / (Config.h1 * Config.h2)],
                [lambda _p: Config.c(_p.x, _p.y) / (Config.h2 ** 2),
                 lambda _p: -(2 * Config.a(_p.x, _p.y) / Config.h1 ** 2 + 2 * Config.c(_p.x, _p.y) / Config.h2 ** 2),
                 lambda _p: Config.c(_p.x, _p.y) / (Config.h2 ** 2)],
                [lambda _p: -2 * Config.b(_p.x, _p.y) / (Config.h1 * Config.h2),
                 lambda _p: Config.a(_p.x, _p.y) / Config.h1 ** 2 + Config.d(_p.x, _p.y) / Config.h1,
                 lambda _p: 2 * Config.b(_p.x, _p.y) / (Config.h1 * Config.h2)]
            ]
            for _ii, delta_i in enumerate([-1, 0, 1]):
                for _ij, delta_j in enumerate([-1, 0, 1]):
                    self.matrix_a[id_num][
                        self.point_dict[f"{point.i + delta_i},{point.j + delta_j}"].id_num
                    ] = schema_list[_ii][_ij](point)

    def solve(self):
        for coordinate, point in tqdm(self.point_dict.items(), desc="正在更新A矩阵..."):
            self.update(point)
        Util.print("正在求A矩阵的逆...")
        self.matrix_u = np.dot(np.linalg.inv(self.matrix_a), self.matrix_f)
        Util.print("求解完成！")
        for _index, u in enumerate(self.matrix_u.ravel()):
            Point.instance_list[_index].set_value(u)


class Handler(object):
    def __init__(self):
        block_marks = np.zeros(Table.x_list.shape)
        for index_i, i in enumerate(tqdm(Table.x_list, desc="正在将Ω区域锯齿化...")):
            for index_j, j in enumerate(Table.y_list[0]):
                block_marks[index_i][index_j] = int(Table.whether_fill_a_block(i[0], j))
        self.table = Table(block_marks)

    def verify_feasibility(self):
        assert Util.is_int(Config.x_amount) and Util.is_int(Config.y_amount)
        try:
            for point in self.table.point_dict.values():
                a_value = Config.a(point.x, point.y)
                b_value = Config.b(point.x, point.y)
                c_value = Config.c(point.x, point.y)
                assert a_value > 0
                assert b_value > 0
                assert a_value * c_value - (b_value ** 2) > 0
        except AssertionError:
            Util.print('警告！不满足"a>0, b>0, ac-b²>0"')
        else:
            Util.print('检验完成！满足"a>0, b>0, ac-b²>0"')


if __name__ == '__main__':
    handler = Handler()
    handler.verify_feasibility()
    handler.table.plot_block_marks()
    handler.table.solve()
    Point.plot_finally()
    Util.print("完成！")
