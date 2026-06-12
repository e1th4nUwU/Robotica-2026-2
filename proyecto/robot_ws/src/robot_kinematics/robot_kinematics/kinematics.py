#!/usr/bin/env python3
from sympy import *
import matplotlib.pyplot as plt


class Robot():
    def __init__(self,
                 l: tuple = (0.3, 0.3, 0.3),
                 h_base: float = 0.3):
        th1, th2, th3 = symbols("theta_1,theta_2,theta_3")

        # Brazo articulado RRR que alcanza puntos en 3D (x, y, z).
        # Junta 1: yaw alrededor de Z  -> orienta el brazo en el plano XY.
        # Junta 2: pitch alrededor de Y -> inclina la viga magenta (sube/baja).
        # Junta 3: pitch alrededor de Y -> inclina la viga amarilla (sube/baja).
        # h_base eleva el plano de trabajo por encima de la base (columna fija).
        T_0_1 = self.tr_h(z=h_base, alpha=th1)
        T_1_2 = self.tr_h(x=l[0], beta=th2)
        T_2_3 = self.tr_h(x=l[1], beta=th3)
        T_3_p = self.tr_h(x=l[2])

        T_0_p = T_0_1 * T_1_2 * T_2_3 * T_3_p
        T_0_p = simplify(T_0_p)

        # Vector de postura en el espacio 3D: [x, y, z]
        xi_0_p = Matrix([T_0_p[0, 3],
                         T_0_p[1, 3],
                         T_0_p[2, 3]])

        # Jacobiano (3x3): mapea velocidades de junta a velocidad cartesiana
        J = Matrix([[diff(xi_0_p, th1),
                     diff(xi_0_p, th2),
                     diff(xi_0_p, th3)]])
        J_inv = J.inv()

        # Velocidades del E.F. en el espacio cartesiano
        x_dot, y_dot, z_dot = symbols("x_dot, y_dot, z_dot")

        # Polinomio lambda para trayectoria suave
        t = symbols("t")
        a_0, a_1, a_2, a_3, a_4, a_5 = symbols("a_0, a_1, a_2, a_3, a_4, a_5")
        lam = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5
        lam_dot = diff(lam, t)
        lam_dot_dot = diff(lam_dot, t)

        self.th1, self.th2, self.th3 = th1, th2, th3
        self.xi_0_p = xi_0_p
        self.J_inv = J_inv
        self.x_dot, self.y_dot, self.z_dot = x_dot, y_dot, z_dot
        self.a_0, self.a_1, self.a_2, self.a_3, self.a_4, self.a_5 = (
            a_0, a_1, a_2, a_3, a_4, a_5)
        self.t = t
        self.lam, self.lam_dot, self.lam_dot_dot = lam, lam_dot, lam_dot_dot

    def def_tray(self, t_f: float = 2, frec: float = 15,
                 th_i: tuple = (0.0, -1.3, 1.7),
                 xi_f: tuple = (0.5, 0.0, 0.5)):
        # Posicion inicial del E.F.
        xi_i = self.xi_0_p.subs({self.th1: th_i[0],
                                  self.th2: th_i[1],
                                  self.th3: th_i[2]})
        self.dt = 1.0 / frec
        self.muestras = int(t_f * frec) + 1

        # Restricciones del polinomio lambda
        eq1 = self.lam.subs({self.t: 0})
        eq2 = self.lam.subs({self.t: t_f}) - 1
        eq3 = self.lam_dot.subs({self.t: 0})
        eq4 = self.lam_dot.subs({self.t: t_f})
        eq5 = self.lam_dot_dot.subs({self.t: 0})
        eq6 = self.lam_dot_dot.subs({self.t: t_f})
        solutions = solve((eq1, eq2, eq3, eq4, eq5, eq6),
                          (self.a_0, self.a_1, self.a_2,
                           self.a_3, self.a_4, self.a_5))

        lam_s = self.lam.subs(solutions)
        lam_dot_s = self.lam_dot.subs(solutions)
        lam_dot_dot_s = self.lam_dot_dot.subs(solutions)

        xi_f_vec = Matrix([xi_f[0], xi_f[1], xi_f[2]])
        xi_eq = xi_i + (xi_f_vec - xi_i) * lam_s
        xi_dot_eq = (xi_f_vec - xi_i) * lam_dot_s
        xi_dot_dot_eq = (xi_f_vec - xi_i) * lam_dot_dot_s

        t_m = Matrix.zeros(1, self.muestras)
        for i in range(self.muestras):
            t_m[i] = self.dt * i

        xi_m = Matrix.zeros(3, self.muestras)
        xi_dot_m = Matrix.zeros(3, self.muestras)
        xi_dot_dot_m = Matrix.zeros(3, self.muestras)

        for i in range(self.muestras):
            xi_m[:, i] = xi_eq.subs({self.t: t_m[i]})
            xi_dot_m[:, i] = xi_dot_eq.subs({self.t: t_m[i]})
            xi_dot_dot_m[:, i] = xi_dot_dot_eq.subs({self.t: t_m[i]})

        print(xi_m[:, self.muestras - 1])

        # Cinematica inversa diferencial
        th_dot_eq = self.J_inv * Matrix([self.x_dot,
                                         self.y_dot,
                                         self.z_dot])
        th_m = Matrix.zeros(3, self.muestras)
        th_dot_m = Matrix.zeros(3, self.muestras)
        th_dot_dot_m = Matrix.zeros(3, self.muestras)

        th_m[:, 0] = Matrix([th_i[0], th_i[1], th_i[2]])

        for i in range(self.muestras):
            th_dot_m[:, i] = th_dot_eq.subs({
                self.th1: th_m[0, i],
                self.th2: th_m[1, i],
                self.th3: th_m[2, i],
                self.x_dot: xi_dot_m[0, i],
                self.y_dot: xi_dot_m[1, i],
                self.z_dot: xi_dot_m[2, i]})
            th_dot_m[:, i] = th_dot_m[:, i].evalf()
            if i < self.muestras - 1:
                th_m[:, i + 1] = th_m[:, i] + th_dot_m[:, i] * self.dt
            if not (i == 0):
                th_dot_dot_m[:, i - 1] = (
                    (th_dot_m[:, i] - th_dot_m[:, i - 1]) / self.dt)

        self.xi_m = xi_m
        self.xi_dot_m = xi_dot_m
        self.xi_dot_dot_m = xi_dot_dot_m
        self.th_m = th_m
        self.th_dot_m = th_dot_m
        self.th_dot_dot_m = th_dot_dot_m
        self.t_m = t_m

    def imp_tray(self, block=True):
        fig, (x_g, y_g, z_g) = plt.subplots(nrows=1, ncols=3)
        fig.suptitle("Posiciones del efector final")
        x_g.set_title("x")
        y_g.set_title("y")
        z_g.set_title("z")
        x_g.plot(self.t_m.T, self.xi_m[0, :].T, color="RED")
        y_g.plot(self.t_m.T, self.xi_m[1, :].T, color="green")
        z_g.plot(self.t_m.T, self.xi_m[2, :].T, color=(0, 0, 1))
        plt.show(block=block)

    def imp_junt(self, block=True):
        fig, (th1_g, th2_g, th3_g) = plt.subplots(nrows=1, ncols=3)
        fig.suptitle("Posiciones de las juntas")
        th1_g.set_title("th1")
        th2_g.set_title("th2")
        th3_g.set_title("th3")
        th1_g.plot(self.t_m.T, self.th_m[0, :].T, color="RED")
        th2_g.plot(self.t_m.T, self.th_m[1, :].T, color="green")
        th3_g.plot(self.t_m.T, self.th_m[2, :].T, color=(0, 0, 1))
        plt.show(block=block)

    def tr_h(self, x=0, y=0, z=0,
             gamma=0, beta=0, alpha=0):
        t_x = Matrix([[1,          0,           0, x],
                      [0, cos(gamma), -sin(gamma), 0],
                      [0, sin(gamma),  cos(gamma), 0],
                      [0,          0,           0, 1]])
        t_y = Matrix([[ cos(beta),         0, sin(beta), 0],
                      [         0,         1,         0, y],
                      [-sin(beta),         0, cos(beta), 0],
                      [         0,         0,         0, 1]])
        t_z = Matrix([[cos(alpha), -sin(alpha), 0, 0],
                      [sin(alpha),  cos(alpha), 0, 0],
                      [         0,           0, 1, z],
                      [         0,           0, 0, 1]])
        tr = simplify(t_x * t_y * t_z)
        return tr


def main():
    robot = Robot()
    robot.def_tray()
    robot.imp_tray()
    robot.imp_junt()


if __name__ == "__main__":
    main()
