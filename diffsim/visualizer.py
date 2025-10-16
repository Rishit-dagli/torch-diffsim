"""
Visualization utilities using Polyscope

Interactive 3D visualization for simulations via Polyscope. Supports:
- Real-time visualization of tetrahedral meshes
- Playback controls (play/pause, step, reset)
- Display of energy values and material properties
- Offline frame capture
- Volume mesh rendering
"""

import polyscope as ps
import numpy as np


class PolyscopeVisualizer:
    """
    Interactive 3D visualizer for physics simulations using Polyscope

    This class provides real-time visualization of tetrahedral mesh deformation
    during physics simulation. It displays the mesh as a volume with proper shading,
    shows simulation statistics, and provides interactive controls.

    Features:

    - **Interactive controls**: Play/pause, step-by-step execution, reset
    - **Real-time statistics**: Time, step count, energy values
    - **Material display**: Shows Young's modulus and Poisson's ratio
    - **Solver parameters**: Displays time step, gravity, etc.
    - Volume rendering via Polyscope

    Parameters:
        simulator (Simulator): The simulator instance to visualize
        window_name (str): Window title (default: "DiffSim Physics Simulator")

    Attributes:
        simulator (Simulator): Reference to the simulator
        mesh (ps.VolumeMesh): Polyscope volume mesh object
        is_playing (bool): Whether simulation is currently running
        show_wireframe (bool): Whether to show mesh edges

    Example:
        >>> sim = Simulator(mesh, material, solver)
        >>> viz = PolyscopeVisualizer(sim)
        >>> viz.run(max_steps=1000, steps_per_frame=5)
    """

    def __init__(self, simulator, window_name="DiffSim Physics Simulator"):
        """
        Initialize visualizer

        Args:
            simulator: Simulator object
            window_name: window title
        """
        self.simulator = simulator
        self.window_name = window_name

        # Initialize polyscope
        ps.init()
        ps.set_ground_plane_mode("tile")  # Show ground plane as a tile grid
        ps.set_ground_plane_height(0.0)  # Align with y=0
        ps.set_up_dir("y_up")  # Set Y as up direction

        # Register mesh
        self._register_mesh()

        # Animation control
        self.is_playing = False
        self.show_wireframe = True

    def _register_mesh(self):
        """Register the mesh with polyscope (as volume mesh)"""
        # Register as tetrahedral volume mesh for proper visualization
        vertices = self.simulator.positions.cpu().numpy()
        tetrahedra = self.simulator.mesh.tetrahedra.cpu().numpy()

        self.mesh = ps.register_volume_mesh(
            "simulation_mesh", vertices, tetrahedra, enabled=True
        )

        # Add some nice visualization options
        self.mesh.set_color((0.3, 0.6, 0.9))
        self.mesh.set_edge_width(1.0)
        self.mesh.set_material("wax")

    def update(self):
        """Update visualization with current simulation state"""
        vertices = self.simulator.positions.cpu().numpy()
        self.mesh.update_vertex_positions(vertices)

    def run(self, max_steps=None, steps_per_frame=1):
        """
        Run interactive visualization loop

        Args:
            max_steps: maximum number of simulation steps (None for infinite)
            steps_per_frame: number of simulation steps per frame
        """
        step_count = 0

        def callback():
            nonlocal step_count

            # UI controls
            if ps.imgui.Button("Play/Pause"):
                self.is_playing = not self.is_playing

            ps.imgui.SameLine()
            if ps.imgui.Button("Step"):
                self.simulator.step()
                self.update()
                step_count += 1

            ps.imgui.SameLine()
            if ps.imgui.Button("Reset"):
                self.simulator.reset()
                self.update()
                step_count = 0

            # Display info
            ps.imgui.Text(f"Time: {self.simulator.time:.3f} s")
            ps.imgui.Text(f"Steps: {step_count}")
            ps.imgui.Text(f"Vertices: {self.simulator.mesh.num_vertices}")
            ps.imgui.Text(f"Elements: {self.simulator.mesh.num_elements}")

            # Energy display
            ke, ee, ge = self.simulator.compute_energy()
            ps.imgui.Text(f"Kinetic Energy: {ke:.2f} J")
            ps.imgui.Text(f"Elastic Energy: {ee:.2f} J")
            ps.imgui.Text(f"Total Energy: {ke + ee + ge:.2f} J")

            # Material parameters
            ps.imgui.Separator()
            ps.imgui.Text("Material Properties:")
            ps.imgui.Text(f"Young's Modulus: {self.simulator.material.E:.2e} Pa")
            ps.imgui.Text(f"Poisson's Ratio: {self.simulator.material.nu:.3f}")

            # Simulation parameters
            ps.imgui.Separator()
            ps.imgui.Text("Solver Parameters:")
            ps.imgui.Text(f"Time Step: {self.simulator.solver.dt:.4f} s")
            g_val = (
                self.simulator.solver.gravity_value
                if self.simulator.solver.gravity is None
                else self.simulator.solver.gravity[1].item()
            )
            ps.imgui.Text(f"Gravity: {g_val:.2f} m/s²")

            # Run simulation if playing
            if self.is_playing:
                for _ in range(steps_per_frame):
                    if max_steps is None or step_count < max_steps:
                        self.simulator.step()
                        step_count += 1
                self.update()

        # Set callback and show
        ps.set_user_callback(callback)
        ps.show()

    def animate_offline(self, num_steps, output_frames=None):
        """
        Run simulation and optionally save frames

        Args:
            num_steps: number of simulation steps
            output_frames: if not None, save frames to this directory
        """
        for i in range(num_steps):
            self.simulator.step()
            self.update()

            if output_frames is not None and i % 10 == 0:
                ps.screenshot(f"{output_frames}/frame_{i:05d}.png")

            if i % 100 == 0:
                print(f"Step {i}/{num_steps}")
