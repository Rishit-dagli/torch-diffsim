import sys
import os

import torch
import numpy as np
import polyscope as ps
from diffsim import TetrahedralMesh, StableNeoHookean, SemiImplicitSolver, Simulator
import subprocess
from pathlib import Path


def setup_polyscope_square():
    ps.init()
    ps.set_ground_plane_mode("tile")
    ps.set_ground_plane_height(0.0)
    ps.set_up_dir("y_up")
    ps.set_window_size(1080, 1080)
    ps.set_automatically_compute_scene_extents(False)
    ps.set_length_scale(5.0)
    ps.set_bounding_box((-1, -0.5, -1), (1, 3, 1))


def setup_camera_interactively():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mesh_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "tetmesh", "bunny0.msh"
    )
    mesh = TetrahedralMesh.from_file(mesh_path, device=device)

    mesh.vertices *= 2.0
    min_y = mesh.vertices[:, 1].min()
    mesh.vertices[:, 1] -= min_y
    mesh.vertices[:, 1] += 2.0
    mesh._compute_rest_state()

    setup_polyscope_square()

    vertices = mesh.vertices.cpu().numpy()
    tetrahedra = mesh.tetrahedra.cpu().numpy()
    mesh_vis = ps.register_volume_mesh(
        "bunny_preview", vertices, tetrahedra, enabled=True
    )
    mesh_vis.set_color((0.3, 0.6, 0.9))
    mesh_vis.set_edge_width(0.0)
    mesh_vis.set_material("wax")

    ps.set_ground_plane_mode("tile_reflection")
    ps.set_shadow_darkness(0.75)
    ps.set_ground_plane_height_factor(0.0)

    ps.look_at((0.0, 1.5, 3.5), (0.0, 1.0, 0.0))

    print("\n" + "=" * 70)
    print("CAMERA SETUP")
    print("=" * 70)
    print("Adjust the camera view in the window, then close it to continue.")
    print("The same camera view will be used for all 25 material simulations.")
    print("=" * 70 + "\n")

    ps.show()

    camera_json = ps.get_view_as_json()

    camera_file = Path("bunny_camera_view.json")
    with open(camera_file, "w") as f:
        f.write(camera_json)

    print(f"Camera view saved to {camera_file}")

    ps.remove_all_structures()

    return str(camera_file)


def run_simulation_with_material(
    youngs_modulus, poissons_ratio, material_id, camera_file, num_frames=600
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(material_id)
    color = tuple(np.random.rand(3))

    print(
        f"\nMaterial {material_id + 1}/25: E={youngs_modulus:.2e} Pa, ν={poissons_ratio:.3f}, Color={color}"
    )

    with open(camera_file, "r") as f:
        camera_json = f.read()

    output_dir = Path(f"bunny_frames/material_{material_id:02d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "tetmesh", "bunny0.msh"
    )
    mesh = TetrahedralMesh.from_file(mesh_path, device=device)

    mesh.vertices *= 2.0
    min_y = mesh.vertices[:, 1].min()
    mesh.vertices[:, 1] -= min_y
    mesh.vertices[:, 1] += 2.0
    mesh._compute_rest_state()

    material = StableNeoHookean(
        youngs_modulus=youngs_modulus, poissons_ratio=poissons_ratio
    )

    solver = SemiImplicitSolver(
        dt=0.003,
        gravity=-9.8,
        damping=0.998,
        substeps=4,
        collision_method="simplified",
        enable_self_collision=False,
    )

    simulator = Simulator(
        mesh=mesh, material=material, solver=solver, density=1000.0, device=device
    )

    setup_polyscope_square()

    vertices = simulator.positions.cpu().numpy()
    tetrahedra = simulator.mesh.tetrahedra.cpu().numpy()
    mesh_vis = ps.register_volume_mesh("bunny", vertices, tetrahedra, enabled=True)
    mesh_vis.set_color(color)
    mesh_vis.set_edge_width(0.0)
    mesh_vis.set_material("wax")

    ps.set_ground_plane_mode("shadow_only")

    ps.set_view_from_json(camera_json)

    steps_per_frame = 2

    for frame in range(num_frames):

        for _ in range(steps_per_frame):
            simulator.step()

        vertices = simulator.positions.cpu().numpy()
        mesh_vis.update_vertex_positions(vertices)

        screenshot_path = output_dir / f"frame_{frame:04d}.png"
        ps.screenshot(str(screenshot_path), transparent_bg=False)

        if frame % 50 == 0:
            print(f"  Frame {frame}/{num_frames}")

    ps.remove_all_structures()

    print(f"  Saved {num_frames} frames to {output_dir}")

    return output_dir


def create_video_from_frames(frames_dir, output_video, fps=30):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        f"{frames_dir}/frame_*.png",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_video),
    ]

    print(f"Creating video: {output_video}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  Video created successfully")


def main():
    print("=" * 70)
    print("BUNNY SIMULATION WITH 25 DIFFERENT MATERIALS")
    print("=" * 70)

    camera_file = setup_camera_interactively()
    print("\nCamera view saved! Starting simulations...\n")

    materials = []

    youngs_moduli = np.logspace(5, 6, 5)
    poissons_ratios = np.linspace(0.3, 0.45, 5)

    for E in youngs_moduli:
        for nu in poissons_ratios:
            materials.append((E, nu))

    print(f"\nWill simulate {len(materials)} material combinations:")
    print(
        f"  Young's modulus range: {youngs_moduli[0]:.2e} - {youngs_moduli[-1]:.2e} Pa"
    )
    print(
        f"  Poisson's ratio range: {poissons_ratios[0]:.3f} - {poissons_ratios[-1]:.3f}"
    )

    video_dir = Path("bunny_videos")
    video_dir.mkdir(exist_ok=True)

    for i, (E, nu) in enumerate(materials):
        frames_dir = run_simulation_with_material(E, nu, i, camera_file, num_frames=600)

        video_path = video_dir / f"bunny_material_{i:02d}_E{E:.0e}_nu{nu:.3f}.mp4"
        create_video_from_frames(frames_dir, video_path, fps=30)

    print("\n" + "=" * 70)
    print("ALL VIDEOS CREATED SUCCESSFULLY")
    print(f"Videos saved to: {video_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
