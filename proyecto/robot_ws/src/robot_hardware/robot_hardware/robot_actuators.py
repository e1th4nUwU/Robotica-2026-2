#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class NodoHardware(Node):
    def __init__(self):
        super().__init__("nodo_hardware")
        # Publicador del estado actual de las juntas
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        # Suscriptor para posiciones deseadas
        self.j_goal_sub = self.create_subscription(
            JointState, "/joint_states_goals", self.goal_callback, 10)

        self.js_state = JointState()
        self.js_state.name = ["shoulder_joint", "arm_joint", "forearm_joint"]
        self.js_state.position = [0.1, 0.1, 0.1]

        self.js_goal = JointState()
        self.js_goal.name = ["shoulder_joint", "arm_joint", "forearm_joint"]
        self.js_goal.position = [0.1, 0.1, 0.1]

        self.create_timer(0.01, self.hw_callback)

    def goal_callback(self, msg: JointState):
        self.js_goal = msg

    def hw_callback(self):
        self.js_state.position = self.js_goal.position
        self.js_state.header.stamp = self.get_clock().now().to_msg()
        self.js_pub.publish(self.js_state)


def main():
    try:
        rclpy.init()
        nodo_hardware = NodoHardware()
        rclpy.spin(nodo_hardware)
        rclpy.shutdown()
    except KeyboardInterrupt as e:
        print(e)
