#!/usr/bin/env python3
import math
import random
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from pso_interfaces.srv import Evaluate2D

# For offline testing without ROS service:
from pso_nodes.function_service import FUNCTIONS  # sphere_2d, rastrigin_2d, rosenbrock_2d


# ----------------------------------
# Search domains for each function
# ----------------------------------
FUNCTION_DOMAINS = {
    "sphere":    (-5.12, 5.12, -5.12, 5.12),
    "rastrigin": (-5.12, 5.12, -5.12, 5.12),
    "rosenbrock": (-2.048, 2.048, -2.048, 2.048),
}


class PSOController(Node):
    def __init__(
        self,
        function_name: str = "sphere",
        swarm_size: int = 10,
        max_epochs: int = 100,
        w: float = 0.7,
        c1: float = 1.4,
        c2: float = 1.4,
    ):
        super().__init__("pso_controller")

        # --- PSO hyperparameters ---
        self.function_name = function_name.lower()
        self.N = swarm_size
        self.max_epochs = max_epochs
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Bounds and vmax from table
        if self.function_name not in FUNCTION_DOMAINS:
            self.get_logger().warn(
                f"Unknown function '{self.function_name}', using 'sphere' instead"
            )
            self.function_name = "sphere"

        (self.x_min, self.x_max, self.y_min, self.y_max) = FUNCTION_DOMAINS[
            self.function_name
        ]
        interval_x = self.x_max - self.x_min
        interval_y = self.y_max - self.y_min
        # We'll use the same clamp for both components based on max range
        self.vmax = 0.2 * max(interval_x, interval_y)

        # --- ROS communication ---
        self.eval_client = self.create_client(Evaluate2D, "evaluate_function")
        self.get_logger().info("Waiting for evaluate_function service...")
        while not self.eval_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("... still waiting for evaluate_function ...")
        self.get_logger().info("Connected to evaluate_function service.")

        # Publishes all particle positions as [x1, y1, x2, y2, ...]
        self.pos_pub = self.create_publisher(Float32MultiArray, "particle_pos", 10)
        # Publishes [gbest_x, gbest_y, gbest_fit]
        self.gbest_pub = self.create_publisher(Float32MultiArray, "global_best", 10)

        # --- For offline fallback (if service fails) ---
        self.local_functions = FUNCTIONS

    # ------------------------------
    # Service or local evaluation
    # ------------------------------
    def evaluate(self, x: float, y: float) -> float:
        """Evaluate f(x,y) via ROS service; fall back to local function if needed."""
        # Build request
        req = Evaluate2D.Request()
        req.x = float(x)
        req.y = float(y)
        req.function_name = self.function_name

        future = self.eval_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return float(future.result().value)
        else:
            self.get_logger().error(
                "Service call failed, falling back to local function implementation."
            )
            f = self.local_functions[self.function_name]
            return float(f(x, y))

    # ------------------------------
    # Helper: random init inside box
    # ------------------------------
    def random_position(self) -> np.ndarray:
        x = random.uniform(self.x_min, self.x_max)
        y = random.uniform(self.y_min, self.y_max)
        return np.array([x, y], dtype=float)

    def clip_position(self, pos: np.ndarray) -> np.ndarray:
        x = min(max(pos[0], self.x_min), self.x_max)
        y = min(max(pos[1], self.y_min), self.y_max)
        return np.array([x, y], dtype=float)

    def clip_velocity(self, vel: np.ndarray) -> np.ndarray:
        return np.clip(vel, -self.vmax, self.vmax)

    # ------------------------------
    # Main PSO routine
    # ------------------------------
    def run(self):
        # Seed for reproducibility (optional)
        random.seed(42)
        np.random.seed(42)

        # 1. Initialize swarm
        positions = np.zeros((self.N, 2), dtype=float)
        velocities = np.zeros((self.N, 2), dtype=float)
        pbest_pos = np.zeros((self.N, 2), dtype=float)
        pbest_fit = np.zeros(self.N, dtype=float)

        for i in range(self.N):
            positions[i] = self.random_position()
            velocities[i] = np.zeros(2)  # or small random
            fit_i = self.evaluate(positions[i, 0], positions[i, 1])
            pbest_pos[i] = positions[i].copy()
            pbest_fit[i] = fit_i

        # 2. Determine initial global best
        best_idx = int(np.argmin(pbest_fit))
        gbest = pbest_pos[best_idx].copy()
        gbest_fit = float(pbest_fit[best_idx])

        self.get_logger().info(
            f"Initial gbest (epoch 0) {gbest} with fitness {gbest_fit:.6e}"
        )

        # Logs for plotting
        gbest_history: List[float] = []
        inertia_history: List[float] = []
        cognitive_history: List[float] = []
        social_history: List[float] = []

        # Early stopping parameters
        tol_abs = 1e-6        # tiny target fitness
        tol_plateau = 1e-8    # how "flat" the min must be
        patience = 10         # epochs to check plateau

        # 3. Main loop
        for epoch in range(1, self.max_epochs + 1):
            inertia_mag_sum = 0.0
            cognitive_mag_sum = 0.0
            social_mag_sum = 0.0

            for i in range(self.N):
                # Random factors
                r1 = random.random()
                r2 = random.random()

                # Velocity components (2D vectors)
                inertia = self.w * velocities[i]
                cognitive = self.c1 * r1 * (pbest_pos[i] - positions[i])
                social = self.c2 * r2 * (gbest - positions[i])

                new_vel = inertia + cognitive + social
                new_vel = self.clip_velocity(new_vel)
                new_pos = positions[i] + new_vel
                new_pos = self.clip_position(new_pos)

                # Evaluate new fitness
                new_fit = self.evaluate(new_pos[0], new_pos[1])

                # Update personal best
                if new_fit < pbest_fit[i]:
                    pbest_fit[i] = new_fit
                    pbest_pos[i] = new_pos.copy()

                # Store for next iteration
                velocities[i] = new_vel
                positions[i] = new_pos

                # Accumulate magnitudes for logging
                inertia_mag_sum += np.linalg.norm(inertia)
                cognitive_mag_sum += np.linalg.norm(cognitive)
                social_mag_sum += np.linalg.norm(social)

            # Update global best
            best_idx = int(np.argmin(pbest_fit))
            if pbest_fit[best_idx] < gbest_fit:
                gbest = pbest_pos[best_idx].copy()
                gbest_fit = float(pbest_fit[best_idx])

            # Average magnitudes across swarm
            inertia_avg = inertia_mag_sum / self.N
            cognitive_avg = cognitive_mag_sum / self.N
            social_avg = social_mag_sum / self.N

            # Log for plotting
            gbest_history.append(gbest_fit)
            inertia_history.append(inertia_avg)
            cognitive_history.append(cognitive_avg)
            social_history.append(social_avg)

            # ROS topics for visualization
            self.publish_positions(positions)
            self.publish_global_best(gbest, gbest_fit)

            self.get_logger().info(
                f"[{self.function_name}] "
                f"epoch {epoch:3d}: gbest_fit = {gbest_fit:.6e}, "
                f"inertia = {inertia_avg:.3e}, "
                f"cog = {cognitive_avg:.3e}, "
                f"social = {social_avg:.3e}"
            )

            # Early stopping: fitness tiny AND plateaued
            if gbest_fit < tol_abs and len(gbest_history) > patience:
                recent = gbest_history[-patience:]
                if max(recent) - min(recent) < tol_plateau:
                    self.get_logger().info(
                        f"Early stopping at epoch {epoch}, "
                        f"gbest_fit ~ {gbest_fit:.6e}"
                    )
                    break

        # After loop: plot results
        self.plot_results(gbest_history, inertia_history, cognitive_history, social_history)

    # ------------------------------
    # ROS publishers
    # ------------------------------
    def publish_positions(self, positions: np.ndarray):
        """Publish all particle positions as [x1, y1, x2, y2, ...]."""
        msg = Float32MultiArray()
        flat = positions.flatten().astype(np.float32)
        msg.data = flat.tolist()
        self.pos_pub.publish(msg)

    def publish_global_best(self, gbest: np.ndarray, gbest_fit: float):
        """Publish [gbest_x, gbest_y, gbest_fit]."""
        msg = Float32MultiArray()
        msg.data = [
            float(gbest[0]),
            float(gbest[1]),
            float(gbest_fit),
        ]
        self.gbest_pub.publish(msg)

    # ------------------------------
    # Plotting with matplotlib
    # ------------------------------
    def plot_results(
        self,
        gbest_history: List[float],
        inertia_history: List[float],
        cognitive_history: List[float],
        social_history: List[float],
    ):
        epochs = np.arange(1, len(gbest_history) + 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, gbest_history)
        plt.xlabel("Epoch")
        plt.ylabel("Global best fitness")
        plt.title(f"PSO on {self.function_name} – gbest fitness")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, inertia_history, label="inertia")
        plt.plot(epochs, cognitive_history, label="cognitive")
        plt.plot(epochs, social_history, label="social")
        plt.xlabel("Epoch")
        plt.ylabel("Average component magnitude")
        plt.title("Velocity component magnitudes")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    # choose function here: "sphere", "rastrigin", or "rosenbrock"
    node = PSOController(function_name="rosenbrock", swarm_size=10, max_epochs=100)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
