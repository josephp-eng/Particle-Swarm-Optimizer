#!/usr/bin/env python3

# Units Under Test (Functions)
# Joey P
# 12/12/25

import math

import rclpy
from rclpy.node import Node

from pso_interfaces.srv import Evaluate2D  # custom service


# -------------------------------
# Benchmark functions
# -------------------------------

def sphere_2d(x: float, y: float) -> float:
    """2-D Sphere function: f(x, y) = x² + y²"""
    return x**2 + y**2


def rastrigin_2d(x: float, y: float) -> float:
    """2-D Rastrigin function.
    f(x, y) = 2*A + (x² - A*cos(2πx)) + (y² - A*cos(2πy))
    """
    A = 10.0
    return (
        2 * A
        + (x**2 - A * math.cos(2 * math.pi * x))
        + (y**2 - A * math.cos(2 * math.pi * y))
    )


def rosenbrock_2d(x: float, y: float) -> float:
    """2-D Rosenbrock function.
    f(x, y) = (a - x)² + b*(y - x²)²    (a=1, b=100)
    """
    a = 1.0
    b = 100.0
    return (a - x)**2 + b * (y - x**2)**2


FUNCTIONS = {
    "sphere": sphere_2d,
    "rastrigin": rastrigin_2d,
    "rosenbrock": rosenbrock_2d,
}


# -------------------------------
# ROS 2 service node
# -------------------------------

class FunctionServiceNode(Node):
    def __init__(self):
        super().__init__("function_service")
        self.srv = self.create_service(
            Evaluate2D,
            "evaluate_function",
            self.evaluate_callback,
        )
        self.get_logger().info("Function service ready: sphere, rastrigin, rosenbrock")

    def evaluate_callback(self, request, response):
        func_name = request.function_name.strip().lower()
        x = request.x
        y = request.y

        if func_name not in FUNCTIONS:
            self.get_logger().warn(
                f"Unknown function '{func_name}', falling back to 'sphere'"
            )
            func_name = "sphere"

        f = FUNCTIONS[func_name]
        value = f(x, y)
        response.value = float(value)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FunctionServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
