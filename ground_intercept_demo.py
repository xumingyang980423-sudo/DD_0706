import argparse
import asyncio
import math
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCone, VisualCuboid, VisualCylinder, VisualSphere
from intercept_core import (
    InterceptScenario,
    MISSILE_VISUAL_NOSE_OFFSET,
    aircraft_quat,
    missile_quat,
    norm,
    quat_from_matrix,
    quat_multiply,
    rotate_vector,
    unit,
)
import omni.kit.viewport.utility
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade


SCENARIO_CHOICES = [
    "head_on",
    "overfly_tail_chase",
    "crossing_left_to_right",
    "crossing_right_to_left",
    "climb_escape",
    "dive_escape",
    "s_turn_evasion",
    "double_evasion",
    "late_launch",
    "high_speed_pass",
    "low_altitude_pass",
    "far_tail_chase",
    "fighter_weave_chase",
    "maneuver_follow_chase",
    "long_weave_tail_chase",
    "extended_maneuver_follow",
    "climb_dive_weave_chase",
    "delayed_sustained_evasion",
    "cobra_pop_up_chase",
    "circle_turn_chase",
    "spiral_climb_chase",
    "hard_reversal_chase",
    "wide_snake_chase",
    "super_combo_chase",
]


class CompoundModel:
    def __init__(self, parts: list) -> None:
        self.parts = parts

    def set_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        for part, local_pos, local_quat in self.parts:
            world_position = position + rotate_vector(orientation, local_pos)
            world_orientation = quat_multiply(orientation, local_quat)
            part.set_world_pose(position=world_position, orientation=world_orientation)


def parse_vec3(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values, for example: 1,1,1")
    return np.array(values, dtype=float)


def euler_xyz_degrees_to_quat(degrees: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.radians(degrees)
    cx, sx = math.cos(rx / 2.0), math.sin(rx / 2.0)
    cy, sy = math.cos(ry / 2.0), math.sin(ry / 2.0)
    cz, sz = math.cos(rz / 2.0), math.sin(rz / 2.0)
    qx = np.array([cx, sx, 0.0, 0.0], dtype=float)
    qy = np.array([cy, 0.0, sy, 0.0], dtype=float)
    qz = np.array([cz, 0.0, 0.0, sz], dtype=float)
    return quat_multiply(quat_multiply(qz, qy), qx)


def local_axis_alignment_quat(
    model_forward: np.ndarray,
    model_up: np.ndarray,
    sim_forward: np.ndarray,
    sim_up: np.ndarray,
) -> np.ndarray:
    model_fwd = unit(model_forward)
    model_right = unit(np.cross(model_up, model_fwd), np.array([0.0, 1.0, 0.0]))
    model_up_orth = unit(np.cross(model_fwd, model_right), model_up)
    model_basis = np.column_stack((model_fwd, model_right, model_up_orth))

    sim_fwd = unit(sim_forward)
    sim_right = unit(np.cross(sim_up, sim_fwd), np.array([0.0, 1.0, 0.0]))
    sim_up_orth = unit(np.cross(sim_fwd, sim_right), sim_up)
    sim_basis = np.column_stack((sim_fwd, sim_right, sim_up_orth))

    return quat_from_matrix(sim_basis @ model_basis.T)


def to_usd_asset_path(path: str) -> str:
    return str(Path(path).expanduser().resolve()).replace("\\", "/")


async def convert_asset_to_usd_async(input_path: Path, output_path: Path) -> bool:
    import omni.kit.asset_converter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_cameras = True
    converter_context.ignore_animation = True
    converter_context.use_meter_as_world_unit = True
    converter_context.create_world_as_default_root_prim = False
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        str(input_path),
        str(output_path),
        lambda progress, total_steps: None,
        converter_context,
    )
    return bool(await task.wait_until_finished())


def prepare_model_asset(path: str | None, label: str) -> str | None:
    if not path:
        return None
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"{label} model not found: {input_path}")

    if input_path.suffix.lower() in {".usd", ".usda", ".usdc"}:
        return to_usd_asset_path(str(input_path))

    output_path = Path("assets") / "converted" / f"{input_path.stem}.usd"
    output_path = output_path.resolve()
    if not output_path.exists() or output_path.stat().st_mtime < input_path.stat().st_mtime:
        print(f"Converting {label} model to USD: {input_path} -> {output_path}", flush=True)
        try:
            success = asyncio.get_event_loop().run_until_complete(convert_asset_to_usd_async(input_path, output_path))
        except RuntimeError:
            success = asyncio.run(convert_asset_to_usd_async(input_path, output_path))
        if not success:
            print(f"WARNING: USD conversion failed for {input_path}; trying direct reference.", flush=True)
            return to_usd_asset_path(str(input_path))
    return to_usd_asset_path(str(output_path))


def create_preview_material(stage, prim_path: str, color: np.ndarray) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.08)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def force_model_color(stage, prim, color: np.ndarray, material_name: str) -> None:
    material = create_preview_material(stage, f"/World/Materials/{material_name}", color)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    for descendant in Usd.PrimRange(prim):
        if descendant.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI.Apply(descendant).Bind(material, UsdShade.Tokens.strongerThanDescendants)


class AssetReferenceModel:
    def __init__(
        self,
        prim_path: str,
        asset_path: str,
        position: np.ndarray,
        orientation: np.ndarray,
        scale: np.ndarray,
        local_rotation_degrees: np.ndarray,
        axis_alignment_quat: np.ndarray,
        color: np.ndarray | None,
        material_name: str,
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        xform = UsdGeom.Xform.Define(stage, prim_path)
        self.prim = xform.GetPrim()
        model_prim = stage.DefinePrim(f"{prim_path}/Model", "Xform")
        model_prim.GetReferences().AddReference(asset_path)
        if color is not None:
            force_model_color(stage, model_prim, color, material_name)
        self.translate_op = xform.AddTranslateOp()
        self.orient_op = xform.AddOrientOp()
        self.scale_op = xform.AddScaleOp()
        self.scale_op.Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
        self.local_quat = quat_multiply(axis_alignment_quat, euler_xyz_degrees_to_quat(local_rotation_degrees))
        self.set_pose(position, orientation)

    def set_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        world_orientation = quat_multiply(orientation, self.local_quat)
        self.translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        self.orient_op.Set(
            Gf.Quatf(
                float(world_orientation[0]),
                float(world_orientation[1]),
                float(world_orientation[2]),
                float(world_orientation[3]),
            )
        )


def add_aircraft(world: World, position: np.ndarray, orientation: np.ndarray) -> CompoundModel:
    blue = np.array([0.05, 0.20, 1.0])
    grey = np.array([0.75, 0.78, 0.82])
    parts = []
    parts.append(
        (
            world.scene.add(
                VisualCuboid(
                    prim_path="/World/Aircraft/Fuselage",
                    name="aircraft_fuselage",
                    position=position,
                    orientation=orientation,
                    size=1.0,
                    scale=np.array([7.0, 0.9, 0.9]),
                    color=grey,
                )
            ),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )
    parts.append(
        (
            world.scene.add(
                VisualCuboid(
                    prim_path="/World/Aircraft/MainWing",
                    name="aircraft_main_wing",
                    position=position,
                    orientation=orientation,
                    size=1.0,
                    scale=np.array([1.3, 10.0, 0.25]),
                    color=blue,
                )
            ),
            np.array([-0.8, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )
    parts.append(
        (
            world.scene.add(
                VisualCuboid(
                    prim_path="/World/Aircraft/TailWing",
                    name="aircraft_tail_wing",
                    position=position,
                    orientation=orientation,
                    size=1.0,
                    scale=np.array([0.9, 4.6, 0.2]),
                    color=blue,
                )
            ),
            np.array([-4.2, 0.0, 0.4]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )
    return CompoundModel(parts)


def add_missile(world: World, position: np.ndarray, orientation: np.ndarray) -> CompoundModel:
    yellow = np.array([1.0, 0.72, 0.05])
    black = np.array([0.02, 0.02, 0.02])
    red = np.array([1.0, 0.05, 0.02])
    parts = []
    parts.append(
        (
            world.scene.add(
                VisualCylinder(
                    prim_path="/World/Missile/Body",
                    name="missile_body",
                    position=position,
                    orientation=orientation,
                    radius=0.22,
                    height=4.0,
                    color=yellow,
                )
            ),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )
    parts.append(
        (
            world.scene.add(
                VisualCone(
                    prim_path="/World/Missile/Nose",
                    name="missile_nose",
                    position=position + np.array([0.0, 0.0, MISSILE_VISUAL_NOSE_OFFSET]),
                    orientation=orientation,
                    radius=0.24,
                    height=0.7,
                    color=red,
                )
            ),
            np.array([0.0, 0.0, MISSILE_VISUAL_NOSE_OFFSET]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )
    for idx, offset in enumerate(
        [
            np.array([0.38, 0.0, -1.75]),
            np.array([-0.38, 0.0, -1.75]),
            np.array([0.0, 0.38, -1.75]),
            np.array([0.0, -0.38, -1.75]),
        ]
    ):
        parts.append(
            (
                world.scene.add(
                    VisualCuboid(
                        prim_path=f"/World/Missile/Fin{idx}",
                        name=f"missile_fin_{idx}",
                        position=position + offset,
                        orientation=orientation,
                        size=1.0,
                        scale=np.array([0.58, 0.06, 0.34]),
                        color=black,
                    )
                ),
                offset,
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
    return CompoundModel(parts)


def add_trail_dot(world: World, prefix: str, index: int, position: np.ndarray, color: np.ndarray) -> None:
    world.scene.add(
        VisualSphere(
            prim_path=f"/World/{prefix}Trail/{prefix}_{index:04d}",
            name=f"{prefix}_trail_{index:04d}",
            position=position,
            radius=0.22 if prefix == "Missile" else 0.28,
            color=color,
        )
    )


def setup_lighting_and_camera() -> None:
    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(650.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(4200.0)
    sun.CreateAngleAttr(0.45)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 15.0, 35.0))

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.AddTranslateOp().Set(Gf.Vec3d(145.0, -245.0, 135.0))
    camera.AddRotateXYZOp().Set(Gf.Vec3f(60.0, 0.0, 30.0))
    camera.CreateFocalLengthAttr(18.0)

    viewport = omni.kit.viewport.utility.get_active_viewport()
    if viewport is not None:
        viewport.camera_path = "/World/Camera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="head_on")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--hold-frames", type=int, default=1800)
    parser.add_argument("--aircraft-model", default=None, help="Optional aircraft asset path, such as FBX/USD.")
    parser.add_argument("--missile-model", default=None, help="Optional missile asset path, such as FBX/USD.")
    parser.add_argument("--aircraft-model-scale", type=parse_vec3, default=np.array([1.0, 1.0, 1.0]))
    parser.add_argument("--missile-model-scale", type=parse_vec3, default=np.array([1.0, 1.0, 1.0]))
    parser.add_argument("--aircraft-model-rotation", type=parse_vec3, default=np.array([0.0, 0.0, 0.0]))
    parser.add_argument("--missile-model-rotation", type=parse_vec3, default=np.array([0.0, 0.0, 0.0]))
    parser.add_argument("--aircraft-model-forward", type=parse_vec3, default=np.array([0.0, 0.0, -1.0]))
    parser.add_argument("--aircraft-model-up", type=parse_vec3, default=np.array([0.0, 1.0, 0.0]))
    parser.add_argument("--missile-model-forward", type=parse_vec3, default=np.array([-1.0, 0.0, 0.0]))
    parser.add_argument("--missile-model-up", type=parse_vec3, default=np.array([0.0, 0.0, 1.0]))
    parser.add_argument("--aircraft-model-color", type=parse_vec3, default=np.array([0.05, 0.25, 1.0]))
    parser.add_argument("--missile-model-color", type=parse_vec3, default=np.array([1.0, 0.08, 0.02]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Starting Isaac Sim two-phase baseline demo: "
        f"scenario={args.scenario}, randomize={args.randomize}, seed={args.seed}",
        flush=True,
    )
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    setup_lighting_and_camera()

    scenario = InterceptScenario(seed=args.seed)
    scenario.reset(randomize=args.randomize, scenario_type=args.scenario)
    aircraft = scenario.aircraft
    missile = scenario.missile

    aircraft_asset = prepare_model_asset(args.aircraft_model, "aircraft")
    missile_asset = prepare_model_asset(args.missile_model, "missile")
    aircraft_axis_quat = local_axis_alignment_quat(
        args.aircraft_model_forward,
        args.aircraft_model_up,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    missile_axis_quat = local_axis_alignment_quat(
        args.missile_model_forward,
        args.missile_model_up,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    )

    if aircraft_asset:
        aircraft_model = AssetReferenceModel(
            "/World/AircraftAsset",
            aircraft_asset,
            aircraft.position,
            aircraft_quat(aircraft.velocity, aircraft.bank),
            args.aircraft_model_scale,
            args.aircraft_model_rotation,
            aircraft_axis_quat,
            args.aircraft_model_color,
            "ExternalAircraftBlue",
        )
    else:
        aircraft_model = add_aircraft(world, aircraft.position, aircraft_quat(aircraft.velocity, aircraft.bank))

    if missile_asset:
        missile_model = AssetReferenceModel(
            "/World/MissileAsset",
            missile_asset,
            missile.position,
            missile_quat(missile.velocity),
            args.missile_model_scale,
            args.missile_model_rotation,
            missile_axis_quat,
            args.missile_model_color,
            "ExternalMissileRed",
        )
    else:
        missile_model = add_missile(world, missile.position, missile_quat(missile.velocity))

    world.scene.add(
        VisualCuboid(
            prim_path="/World/Launcher/Base",
            name="launcher_base",
            position=np.array([0.0, 0.0, 0.3]),
            size=1.0,
            scale=np.array([6.0, 6.0, 0.6]),
            color=np.array([0.25, 0.27, 0.28]),
        )
    )

    world.reset()

    trail_idx = 0
    hit = False

    while simulation_app.is_running():
        _, _, terminated, truncated, info = scenario.step(use_baseline=True)
        aircraft = scenario.aircraft
        missile = scenario.missile

        aircraft_model.set_pose(aircraft.position, aircraft_quat(aircraft.velocity, aircraft.bank))
        missile_model.set_pose(missile.position, missile_quat(missile.velocity))

        if trail_idx % 5 == 0:
            add_trail_dot(world, "Missile", trail_idx, missile.position.copy(), np.array([1.0, 0.05, 0.02]))
            add_trail_dot(world, "Aircraft", trail_idx, aircraft.position.copy(), np.array([0.05, 0.25, 1.0]))

        if terminated or truncated:
            hit = info["status"] == "hit"
            print(
                f"{info['status'].upper()}: phase={info['phase']}, "
                f"closest_distance={info['closest_distance']:.2f} m, sim_time={info['time']:.2f} s",
                flush=True,
            )
            csv_path = f"logs/baseline_visual_{args.scenario}.csv"
            scenario.write_records_csv(csv_path)
            for _ in range(args.hold_frames):
                world.step(render=True)
            break

        world.step(render=True)
        trail_idx += 1

    if hit:
        print(f"Baseline visual episode data written to logs/baseline_visual_{args.scenario}.csv", flush=True)
    print("Closing Isaac Sim two-phase baseline demo.", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
