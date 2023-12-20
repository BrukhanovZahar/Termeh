import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import sympy as sp

"""Определение параметров"""
a = 10  # Длина маятника
lAB = 5  # Длина линии AB
sA = 1  # Размер объекта A
t = sp.Symbol('t')  # Время как символьная переменная

# Определение s и phi
xA = 4 * sp.cos(3 * t)
phi = 4 * sp.sin(t - 10)

# Движение объекта B
xB = xA + lAB * sp.sin(phi)
yB = lAB * sp.cos(phi)

# Модули скорости и ускорения объекта B
VmodB = sp.sqrt(sp.diff(xB, t) ** 2 + sp.diff(yB, t) ** 2)
WmodB = sp.sqrt(sp.diff(xB, t, 2) ** 2 + sp.diff(yB, t, 2) ** 2)

"""Построение функций"""
countOfFrames = 200
T_start, T_stop = 0, 12
T = np.linspace(T_start, T_stop, countOfFrames)

# Лямбда-функции для численных значений
XA_def = sp.lambdify(t, xA)
XB_def = sp.lambdify(t, xB)
YB_def = sp.lambdify(t, yB)
VmodB_def = sp.lambdify(t, VmodB)
WmodB_def = sp.lambdify(t, WmodB)

XA = XA_def(T)
XB = XB_def(T)
YB = YB_def(T)
VB = VmodB_def(T)
WB = WmodB_def(T)

"""Построение графика"""

fig = plt.figure(figsize=(10, 7))

# Один подграфик на всю ширину
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(ylim=[-a, XA.max() + a], xlim=[min(-lAB, -a), max(lAB, a)])
ax1.set_xlabel('ось y')
ax1.set_ylabel('ось x')
ax1.invert_yaxis()

# Исходные точки D и E
ax1.plot(-a, XA.min(), marker='o', color='black')
ax1.plot(a, XA.min(), marker='o', color='black')

# Линии, между которыми находится A
ax1.plot([-sA / 2, -sA / 2], [XA.min() - sA, XA.max() + sA], linestyle='-.', color='black')
ax1.plot([sA / 2, sA / 2], [XA.min() - sA, XA.max() + sA], linestyle='-.', color='black')

# Начальные положения

# A
PA = ax1.add_patch(Rectangle(xy=[-sA / 2, XA[0] - sA / 2], width=sA, height=sA, color='green'))

# B
PB, = ax1.plot(YB[0], XB[0], marker='o', color='r', markersize=10)

# Линия AB
PAB, = ax1.plot([0, YB[0]], [XA[0], XB[0]], 'black')

# Линии DA и EA
PDA, = ax1.plot([-a, 0], [XA.min(), XA[0]], linestyle='--', color='m')
PEA, = ax1.plot([a, 0], [XA.min(), XA[0]], linestyle='--', color='m')


# Функция для обновления положения
def anima(i):
    PA.set(xy=[-sA / 2, XA[i] - sA / 2])
    PB.set_data(YB[i], XB[i])
    PAB.set_data([0, YB[i]], [XA[i], XB[i]])
    PDA.set_data([-a, 0], [XA.min(), XA[i]])
    PEA.set_data([a, 0], [XA.min(), XA[i]])
    return PAB, PDA, PEA, PA, PB

# Анимационная функция
anim = FuncAnimation(fig, anima, frames=countOfFrames, interval=100, blit=True)

plt.show()
