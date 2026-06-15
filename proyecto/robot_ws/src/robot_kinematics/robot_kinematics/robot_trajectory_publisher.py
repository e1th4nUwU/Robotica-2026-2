#!/usr/bin/env python3
import rclpy
import matplotlib.pyplot as plt
from rclpy.node import Node
from robot_kinematics.kinematics import Robot
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import JointState


class PublicadorTrayectoria(Node):
    def __init__(self):
        super().__init__("nodo_publicador")
        self.robot = Robot()

        # Altura del plano de trabajo elevado por encima de la base.
        # El click en RViz da (x, y); la z se fija a este plano para que
        # el efector final alcance un punto XYZ y las vigas se inclinen.
        # Debe ser alcanzable: el robot llega hasta z=0.9 m.
        self.plano_z = 0.6

        # Suscriptor para posiciones deseadas (Twist)
        self.sub_twist = self.create_subscription(
            Twist, "/goals_twist", self.twist_callback, 1)

        # Suscriptor para posiciones deseadas por click en RViz (PointStamped)
        self.sub_ps = self.create_subscription(
            PointStamped, "/clicked_point", self.ps_callback, 1)

        # Publicador de estado deseado de las juntas
        self.js_pub = self.create_publisher(JointState, "/joint_states_goals", 1)

        # Suscriptor al estado actual de las juntas
        self.js_sub = self.create_subscription(
            JointState, "/joint_states", self.js_callback, 10)

        self.is_moving = False

        self.joint_state_msg = JointState()
        self.joint_state_msg.name = ["shoulder_joint",
                                     "arm_joint",
                                     "forearm_joint"]

    def twist_callback(self, msg: Twist):
        if self.is_moving:
            return
        self.is_moving = True
        self.get_logger().info("Posicion recibida: {}".format(str(msg.linear)))
        self.robot.def_tray(
            th_i=(self.js_current.position[0],
                  self.js_current.position[1],
                  self.js_current.position[2]),
            xi_f=(msg.linear.x,
                  msg.linear.y,
                  msg.linear.z))
        self._iniciar_publicacion()

    def ps_callback(self, msg: PointStamped):
        if self.is_moving:
            return
        self.is_moving = True
        x_f = msg.point.x
        y_f = msg.point.y
        # El click solo da X,Y (RViz proyecta sobre el suelo z=0); la Z se
        # fija al plano de trabajo elevado para que el efector final alcance
        # el punto XYZ inclinando las vigas.
        z_f = self.plano_z
        self.get_logger().info(
            "Click recibido: x={:.3f}, y={:.3f}, z={:.3f}".format(
                x_f, y_f, z_f))
        self.robot.def_tray(
            t_f=3, frec=20,
            th_i=(self.js_current.position[0],
                  self.js_current.position[1],
                  self.js_current.position[2]),
            xi_f=(x_f, y_f, z_f))
        self._iniciar_publicacion()

    def _iniciar_publicacion(self):
        # EF REAL: cinematica directa de las juntas finales (no la
        # trayectoria deseada). Asi se ve si el brazo de verdad llego.
        n = self.robot.muestras - 1
        th_f = self.robot.th_m[:, n]
        ef_real = self.robot.fk_num(
            float(th_f[0]), float(th_f[1]), float(th_f[2])).flatten()
        objetivo = self.robot.xi_m[:, n]
        gap = float(((ef_real - objetivo) ** 2).sum() ** 0.5)
        self.get_logger().info("EF objetivo: {}".format(objetivo))
        self.get_logger().info("EF alcanzado (FK real): {}".format(ef_real))
        if gap > 0.01:
            self.get_logger().warn(
                "Punto fuera de alcance: el brazo quedo a {:.3f} m del "
                "objetivo (se estiro lo maximo posible).".format(gap))
        self.get_logger().info("Posicion final juntas: {}".format(th_f))
        # Popup con las graficas al terminar de calcular la trayectoria.
        # Cierra las anteriores y muestra sin bloquear, para que el robot
        # se mueva mientras las ventanas quedan visibles.
        plt.close("all")
        self.robot.imp_tray(block=False)
        self.robot.imp_junt(block=False)
        plt.pause(0.001)
        self.current_pos = 0
        self.timer_pub = self.create_timer(self.robot.dt, self.timer_pub_callback)

    def timer_pub_callback(self):
        self.joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_state_msg.position = [
            float(self.robot.th_m[0, self.current_pos]),
            float(self.robot.th_m[1, self.current_pos]),
            float(self.robot.th_m[2, self.current_pos])]
        self.js_pub.publish(self.joint_state_msg)
        # Mantiene vivas las ventanas de las graficas mientras el robot se mueve
        plt.pause(0.001)
        self.current_pos += 1
        if self.current_pos == (self.robot.muestras - 1):
            self.is_moving = False
            self.timer_pub.destroy()

    def js_callback(self, msg: JointState):
        self.js_current = msg


def main():
    try:
        rclpy.init()
        publicador = PublicadorTrayectoria()
        rclpy.spin(publicador)
        rclpy.shutdown()
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()
