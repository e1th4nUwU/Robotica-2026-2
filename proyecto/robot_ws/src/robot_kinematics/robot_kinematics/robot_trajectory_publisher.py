#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from robot_kinematics.kinematics import Robot
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import JointState


class PublicadorTrayectoria(Node):
    def __init__(self):
        super().__init__("nodo_publicador")
        self.robot = Robot()

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
                  msg.angular.z))
        self._iniciar_publicacion()

    def ps_callback(self, msg: PointStamped):
        if self.is_moving:
            return
        self.is_moving = True
        x_f = msg.point.x
        y_f = msg.point.y
        # Beta apunta el efector final hacia el objetivo para que la barra
        # amarilla (forearm) rote naturalmente al cambiar de posicion
        beta_f = math.atan2(y_f, x_f)
        self.get_logger().info(
            "Click recibido: x={:.3f}, y={:.3f}, beta_f={:.3f}".format(
                x_f, y_f, beta_f))
        self.robot.def_tray(
            t_f=3, frec=20,
            th_i=(self.js_current.position[0],
                  self.js_current.position[1],
                  self.js_current.position[2]),
            xi_f=(x_f, y_f, beta_f))
        self._iniciar_publicacion()

    def _iniciar_publicacion(self):
        self.get_logger().info("Posicion final EF: {}".format(
            self.robot.xi_m[:, self.robot.muestras - 1]))
        self.get_logger().info("Posicion final juntas: {}".format(
            self.robot.th_m[:, self.robot.muestras - 1]))
        self.current_pos = 0
        self.timer_pub = self.create_timer(self.robot.dt, self.timer_pub_callback)

    def timer_pub_callback(self):
        self.joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_state_msg.position = [
            float(self.robot.th_m[0, self.current_pos]),
            float(self.robot.th_m[1, self.current_pos]),
            float(self.robot.th_m[2, self.current_pos])]
        self.js_pub.publish(self.joint_state_msg)
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
