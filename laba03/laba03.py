import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.integrate import odeint
import sympy as sp


# Функция, описывающая систему дифференциальных уравнений
def formY(y, t, fV, fOm):
    y1, y2, y3, y4 = y
    dydt = [y3, y4, fV(y1, y2, y3, y4), fOm(y1, y2, y3, y4)]
    return dydt


# Определение параметров системы
a = 1  # Длина AO = OE = a
lAB = 2  # Длина линии AB
sA = 0.3  # Размер объекта A
mA = 4  # Масса A
mB = 3  # Масса B
g = 9.81  # Ускорение свободного падения
k = 50  # Коэффициент жесткости пружины

# Определение символов и функций SymPy
t = sp.Symbol('t')
xA = sp.Function('x')(t)
phi = sp.Function('phi')(t)
VA = sp.Function('V')(t)
omB = sp.Function('om')(t)

# Построение уравнений Лагранжа
# Выражение для квадрата скорости груза B относительно точки O
VB2 = VA ** 2 + omB ** 2 * lAB ** 2 - 2 * VA * omB * lAB * sp.sin(phi)
# Момент инерции груза B относительно точки O
JB = (mB * lAB ** 2) / 3
# Кинетическая энергия груза B
EkinB = (mB * VB2) / 2 + (JB * omB ** 2) / 2
# Кинетическая энергия массы A
EkinA = (mA * VA ** 2) / 2
# Полная кинетическая энергия системы
Ekin = EkinA + EkinB
# Переменная для выражения деформации пружин
delta_x = sp.sqrt(a ** 2 + xA ** 2) - a
# Потенциальная энергия пружин
EpotStrings = k * delta_x ** 2
# Потенциальная энергия массы A
EpotA = -mA * g * xA
# Потенциальная энергия груза B
EpotB = -mB * g * (xA + lAB * sp.cos(phi))
# Полная потенциальная энергия системы
Epot = EpotStrings + EpotA + EpotB
# Лагранжиан (разность кинетической и потенциальной энергии)
L = Ekin - Epot
# Уравнение Лагранжа для координаты xA
ur1 = sp.diff(sp.diff(L, VA), t) - sp.diff(L, xA)
# Уравнение Лагранжа для угла поворота omB
ur2 = sp.diff(sp.diff(L, omB), t) - sp.diff(L, phi)
# Коэффициенты уравнения для VA
a11 = ur1.coeff(sp.diff(VA, t), 1)
a12 = ur1.coeff(sp.diff(omB, t), 1)
# Коэффициенты уравнения для omB
a21 = ur2.coeff(sp.diff(VA, t), 1)
a22 = ur2.coeff(sp.diff(omB, t), 1)
# Свободные члены уравнений
b1 = -(ur1.coeff(sp.diff(VA, t), 0)).coeff(sp.diff(omB, t), 0).subs([(sp.diff(xA, t), VA), (sp.diff(phi, t), omB)])
b2 = -(ur2.coeff(sp.diff(VA, t), 0)).coeff(sp.diff(omB, t), 0).subs([(sp.diff(xA, t), VA), (sp.diff(phi, t), omB)])
# Определитель матрицы коэффициентов уравнений
det = a11 * a22 - a12 * a21
# Определитель матрицы коэффициентов для VA
det1 = b1 * a22 - b2 * a12
# Определитель матрицы коэффициентов для omB
det2 = a11 * b2 - b1 * a21
# Уравнение для ускорения VA
dVAdt = det1 / det
# Уравнение для ускорения omB
domBdt = det2 / det


# Подготовка данных для численного интегрирования
countOfFrames = 300
y0 = [0, 2, 0, 0]  # x(0), phi(0), v(0), om(0)
T_start, T_stop = 0, 20
T = np.linspace(T_start, T_stop, countOfFrames)

fVA = sp.lambdify([xA, phi, VA, omB], dVAdt, "numpy")
fOmB = sp.lambdify([xA, phi, VA, omB], domBdt, "numpy")
sol = odeint(formY, y0, T, args=(fVA, fOmB))


XA_def = sp.lambdify(xA, xA)
XB_def = sp.lambdify([xA, phi], xA + lAB * sp.cos(phi))
YB_def = sp.lambdify(phi, lAB * sp.sin(phi))

XA = XA_def(sol[:, 0])
XB = XB_def(sol[:, 0], sol[:, 1])
YB = YB_def(sol[:, 1])

dphi = sol[:,3]
RA = mB * g * np.cos(YB) - mB * lAB * dphi ** 2 + k * (XA - a)


fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs1 = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs1.plot(T, XA, color='Blue')
ax_for_graphs1.set_title("x(t)")
ax_for_graphs1.set(xlim=[T_start, T_stop])
ax_for_graphs1.grid(True)

ax_for_graphs2 = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs2.plot(T, YB, color='Red')
ax_for_graphs2.set_title("phi(t)")
ax_for_graphs2.set(xlim=[T_start, T_stop])
ax_for_graphs2.grid(True)

ax_for_graphs3 = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs3.plot(T, RA, color='Black')
ax_for_graphs3.set_title("RA(t)")
ax_for_graphs3.set(xlim=[T_start, T_stop])
ax_for_graphs3.grid(True)


# Построение графиков
fig = plt.figure(figsize=(10, 7))

# График 1: Положение объектов в пространстве
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(ylim=[-a, XA.max() + a], xlim=[min(-lAB, -a), max(lAB, a)])
ax1.set_xlabel('ось y')
ax1.set_ylabel('ось x')
ax1.invert_yaxis()

# Рисование точек D и E
ax1.plot(-a, 0, marker='o', color='black')
ax1.plot(a, 0, marker='o', color='black')

# Рисование линий, между которыми находится A
ax1.plot([-sA / 2, -sA / 2], [XA.min(), XA.max()], linestyle='-.', color='black')
ax1.plot([sA / 2, sA / 2], [XA.min(), XA.max()], linestyle='-.', color='black')

# Рисование начальных положений

# Рисование объекта A
PA = ax1.add_patch(Rectangle(xy=[-sA / 2, XA[0] - sA / 2], width=sA, height=sA, color='g'))

# Рисование объекта B
PB, = ax1.plot(YB[0], XB[0], marker='o', color='r')

# Рисование линии AB
PAB, = ax1.plot([0, YB[0]], [XA[0], XB[0]], 'black')

# Рисование линий DA и EA
PDA, = ax1.plot([-a, 0], [0, XA[0]], linestyle='--', color='m')
PEA, = ax1.plot([a, 0], [0, XA[0]], linestyle='--', color='m')


# Функция для пересчета положений
def anima(i):
    PA.set(xy=[-sA / 2, XA[i] - sA / 2])
    PB.set_data(YB[i], XB[i])
    PAB.set_data([0, YB[i]], [XA[i], XB[i]])
    PDA.set_data([-a, 0], [0, XA[i]])
    PEA.set_data([a, 0], [0, XA[i]])


    return PAB, PDA, PEA, PA, PB


# Функция для анимации
anim = FuncAnimation(fig, anima, frames=countOfFrames, interval=100, blit=True)

plt.show()
