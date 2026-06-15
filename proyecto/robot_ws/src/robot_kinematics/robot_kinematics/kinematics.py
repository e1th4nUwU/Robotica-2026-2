#!/usr/bin/env python3
from sympy import *
import numpy as np
import matplotlib.pyplot as plt


class Robot():
    def __init__(self,
                 l: tuple = (0.3, 0.3, 0.3),
                 h_base: float = 0.3):
        th1, th2, th3 = symbols("theta_1,theta_2,theta_3")

        # Brazo articulado RRR que alcanza puntos en 3D (x, y, z).
        # Junta 1: yaw alrededor de Z  -> el eslabon CYAN gira (orienta el
        #          plano de trabajo en XY); no se inclina.
        # Junta 2: pitch alrededor de Y -> inclina la viga magenta (sube/baja).
        # Junta 3: pitch alrededor de Y -> inclina la viga amarilla (codo).
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

        # Funciones numericas rapidas (lambdify) para la IK por pasos.
        # Evaluar simbolicamente J.inv() en cada muestra era lento y, cerca
        # de singularidades, J.inv() explota. Trabajamos en numpy y usamos
        # minimos cuadrados amortiguados (DLS) en def_tray.
        self.fk_num = lambdify((th1, th2, th3), xi_0_p, "numpy")
        self.J_num = lambdify((th1, th2, th3), J, "numpy")

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
        self.J = J
        self.x_dot, self.y_dot, self.z_dot = x_dot, y_dot, z_dot
        self.a_0, self.a_1, self.a_2, self.a_3, self.a_4, self.a_5 = (
            a_0, a_1, a_2, a_3, a_4, a_5)
        self.t = t
        self.lam, self.lam_dot, self.lam_dot_dot = lam, lam_dot, lam_dot_dot

    def def_tray(self, t_f: float = 2, frec: float = 15,
                 th_i: tuple = (0.0, -1.3, 1.7),
                 xi_f: tuple = (0.5, 0.0, 0.5),
                 k_clik: float = 10.0, lam_dls: float = 0.05):
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

        # Interpolacion en coordenadas CILINDRICAS (r, phi, z), no en linea
        # recta cartesiana. El efector arquea alrededor de la base con radio
        # > 0, asi NO cruza el eje Z (singularidad de yaw) y puede alcanzar
        # puntos detras del robot girando ~180 grados de forma suave.
        p_i = np.array(xi_i, dtype=float).flatten()
        p_f = np.array([xi_f[0], xi_f[1], xi_f[2]], dtype=float)
        r_i = np.hypot(p_i[0], p_i[1])
        r_f = np.hypot(p_f[0], p_f[1])
        phi_i = np.arctan2(p_i[1], p_i[0])
        phi_f = np.arctan2(p_f[1], p_f[0])
        z_i, z_f = p_i[2], p_f[2]
        # Giro de yaw por el CAMINO MAS CORTO (envuelto a (-pi, pi]). La
        # junta de yaw es continua (gira 360 libre), asi que siempre puede
        # tomar el giro minimo hacia cualquier direccion, incluso cruzando
        # la parte de atras, sin atorarse en +-pi.
        dphi = np.arctan2(np.sin(phi_f - phi_i), np.cos(phi_f - phi_i))

        # Funciones numericas del polinomio quintico lambda(t)
        lam_f = lambdify(self.t, lam_s, "numpy")
        lam_dot_f = lambdify(self.t, lam_dot_s, "numpy")
        lam_dot_dot_f = lambdify(self.t, lam_dot_dot_s, "numpy")

        t_m = np.array([self.dt * i for i in range(self.muestras)])

        # Trayectoria deseada (cartesiana) y sus derivadas, via cilindricas.
        xi_m = np.zeros((3, self.muestras))
        xi_dot_m = np.zeros((3, self.muestras))
        xi_dot_dot_m = np.zeros((3, self.muestras))
        for i in range(self.muestras):
            L = float(lam_f(t_m[i]))
            Ld = float(lam_dot_f(t_m[i]))
            Ldd = float(lam_dot_dot_f(t_m[i]))
            rr = r_i + (r_f - r_i) * L
            rd = (r_f - r_i) * Ld
            rdd = (r_f - r_i) * Ldd
            ph = phi_i + dphi * L
            phd = dphi * Ld
            phdd = dphi * Ldd
            zz = z_i + (z_f - z_i) * L
            zd = (z_f - z_i) * Ld
            zdd = (z_f - z_i) * Ldd
            c, s = np.cos(ph), np.sin(ph)
            xi_m[:, i] = [rr * c, rr * s, zz]
            xi_dot_m[:, i] = [rd * c - rr * s * phd,
                              rd * s + rr * c * phd,
                              zd]
            xi_dot_dot_m[:, i] = [
                rdd * c - 2 * rd * s * phd - rr * c * phd ** 2 - rr * s * phdd,
                rdd * s + 2 * rd * c * phd - rr * s * phd ** 2 + rr * c * phdd,
                zdd]

        print("Objetivo cartesiano:", xi_m[:, -1])

        # Cinematica inversa diferencial con realimentacion (CLIK) y
        # minimos cuadrados amortiguados con damping ADAPTATIVO.
        #   v = xi_dot_deseado + K (xi_deseado - FK(theta))   <- corrige deriva
        #   theta_dot = J^T (J J^T + lam^2 I)^-1 v             <- robusto en singularidad
        # El damping crece solo cerca de singularidades (manipulabilidad
        # baja) y es casi nulo lejos de ellas -> buen seguimiento sin las
        # oscilaciones que aparecian con damping fijo en el borde del
        # espacio de trabajo. Ademas se acota el error y se satura la
        # velocidad de junta para evitar zigzag cuando el punto es dificil.
        I3 = np.eye(3)
        w0 = 0.04          # umbral de manipulabilidad para activar damping
        err_max = 0.05     # cota del error cartesiano (m) en el CLIK
        thd_max = 3.0      # saturacion de velocidad de junta (rad/s)
        th_lim = 3.14      # limite articular del URDF (rad)
        th_m = np.zeros((3, self.muestras))
        th_dot_m = np.zeros((3, self.muestras))
        th_dot_dot_m = np.zeros((3, self.muestras))
        th_m[:, 0] = np.array([th_i[0], th_i[1], th_i[2]], dtype=float)

        for i in range(self.muestras):
            th = th_m[:, i]
            p_actual = np.array(
                self.fk_num(th[0], th[1], th[2]), dtype=float).flatten()
            error = xi_m[:, i] - p_actual
            # Acota la magnitud del error para que el termino de realimentacion
            # no dispare la velocidad cuando el objetivo es poco alcanzable.
            e_norm = np.linalg.norm(error)
            if e_norm > err_max:
                error = error * (err_max / e_norm)
            v = xi_dot_m[:, i] + k_clik * error

            J = np.array(self.J_num(th[0], th[1], th[2]), dtype=float)
            # Manipulabilidad w = sqrt(det(J J^T)); damping variable (Nakamura)
            w = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
            if w < w0:
                lam2 = (1.0 - (w / w0) ** 2) * (lam_dls ** 2)
            else:
                lam2 = 0.0
            J_dls = J.T @ np.linalg.inv(J @ J.T + lam2 * I3)
            thd = J_dls @ v
            # Satura la velocidad de junta para mantener curvas suaves.
            thd_peak = np.abs(thd).max()
            if thd_peak > thd_max:
                thd = thd * (thd_max / thd_peak)
            th_dot_m[:, i] = thd
            if i < self.muestras - 1:
                th_next = th_m[:, i] + thd * self.dt
                # El yaw (junta 1) es CONTINUO: no se limita, gira 360 libre.
                # Solo se acotan los pitch (juntas 2 y 3) al rango del URDF.
                th_next[1] = np.clip(th_next[1], -th_lim, th_lim)
                th_next[2] = np.clip(th_next[2], -th_lim, th_lim)
                th_m[:, i + 1] = th_next
            if i != 0:
                th_dot_dot_m[:, i - 1] = (
                    (th_dot_m[:, i] - th_dot_m[:, i - 1]) / self.dt)

        self.xi_m = xi_m
        self.xi_dot_m = xi_dot_m
        self.xi_dot_dot_m = xi_dot_dot_m
        self.th_m = th_m
        self.th_dot_m = th_dot_m
        self.th_dot_dot_m = th_dot_dot_m
        self.t_m = np.array(t_m, dtype=float).reshape(1, -1)

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
