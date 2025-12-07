#!/usr/bin/env python
import numpy as np
import torch
import matplotlib.pyplot as plt

# 네 프로젝트의 모듈 import
from path_command_cfg import PathCommandCfg
from path_command import PathCommand  # 쓰진 않지만 경로 설정 참고용
from poly_spline import PolySplinePath


def select_waypoints_by_interval(lengths, num_waypoints, interval):
    """
    IsaacLab PathCommand._initial_waypoints_indices 와 같은 로직을
    단일 env (1D) 버전으로 단순화한 함수.

    lengths: (num_points,)  누적 거리
    num_waypoints: 뽑을 waypoint 개수 (예: 10)
    interval: 기본 간격 (m)
    """
    # adjust interval if it is larger than the total length
    total_len = lengths[-1]
    if total_len >= num_waypoints * interval:
        adjusted_interval = interval
    else:
        adjusted_interval = total_len / num_waypoints

    # 원하는 거리 지점들
    desired_distances = np.arange(0, num_waypoints) * adjusted_interval

    # searchsorted: 각 desired_distances 에 대해 오른쪽 이웃 index
    search_indices = np.searchsorted(lengths, desired_distances, side="right")
    search_indices = np.clip(search_indices, 0, len(lengths) - 1)

    prev_indices = np.clip(search_indices - 1, 0, len(lengths) - 1)

    left_diff = desired_distances - lengths[prev_indices]
    right_diff = lengths[search_indices] - desired_distances

    indices = np.where(left_diff <= right_diff, prev_indices, search_indices)
    return indices.astype(int)


def ayro_yaw_profile(psi_orig, g, zeta_min=0.35, zeta_max=1.0,
                     omega_min=3.0, omega_max=8.0, blend=True):
    """
    네가 PathCommand._update_command 에서 쓰던 2차 시스템 yaw shaping 로직을
    1D (단일 env) 버전으로 구현한 함수.

    psi_orig : (N,)  원래 yaw 프로파일 (base frame 기준)
    g        : 스칼라 in [0, 1]
    blend    : True 면  (1-g)*psi_orig + g*psi_2nd
               False 면 psi_2nd 만 사용
    """
    psi_orig = np.asarray(psi_orig)
    N = psi_orig.shape[0]

    # goal yaw: 마지막 waypoint yaw
    psi_goal = psi_orig[-1]

    # g → (zeta, omega_n)
    zeta = zeta_max - g * (zeta_max - zeta_min)
    omega_n = omega_min + g * (omega_max - omega_min)

    # pseudo-time 적분
    N_eff = max(N - 1, 1)
    ds = 1.0 / float(N_eff)

    psi = 0.0
    psi_dot = 0.0
    psi_2nd = np.zeros_like(psi_orig)

    for k in range(N):
        psi_2nd[k] = psi
        psi_ddot = -2.0 * zeta * omega_n * psi_dot - (omega_n ** 2) * (psi - psi_goal)
        psi_dot = psi_dot + psi_ddot * ds
        psi = psi + psi_dot * ds

    if blend:
        # g=0 → psi_orig, g=1 → pure 2nd-order
        yaw_profile = (1.0 - g) * psi_orig + g * psi_2nd
    else:
        yaw_profile = psi_2nd

    return yaw_profile, psi_2nd


def sample_path_like_env():
    """
    PathCommandCfg + PolySplinePath 를 이용해서
    네 CommandsCfg(path_command) 와 최대한 비슷한 방식으로 path 하나를 샘플링.
    """
    # 네가 teacher에서 사용한다던 설정을 직접 넣어준다.
    # (필요하면 여기 숫자만 수정해서 테스트)
    max_speed = 7.0
    path_config = {
        "spline_angle_range": (0.0, 120.0),
        "rotate_angle_range": (0.0, 150.0),
        "pos_tolerance_range": (0.2, 0.2),
        "terrain_level_range": (0, 0),
        "resolution": [10.0, 10.0, 0.2, 1],
        "initial_params": [30.0, 40.0, 0.2, 0],
    }

    # PolySplinePath 생성 (path_command.py 에서와 동일)
    res_spline = path_config["resolution"][0]
    res_rotate = path_config["resolution"][1]
    path_generator = PolySplinePath(res_spline, res_rotate)

    # sample_paths 인자 맞춰주기
    # params_list: [angle, rot_angle, pos_tol, terrain_level]
    params_list = [path_config["initial_params"]]
    # num_list: 각 param set 당 path 개수
    num_list = [1]
    # speeds_tensor: 평균 속도 (여기서는 max_speed 하나만 사용)
    speeds_tensor = torch.tensor([max_speed])

    # use_rsl_path=False 로 가정
    paths, param_sets = path_generator.sample_paths(
        params_list, speeds_tensor.tolist(), num_list, use_rsl_path=False
    )

    # paths shape: (num_envs, num_points, 4)  여기서는 num_envs = 1
    # (x, y, yaw, cumulative_length)
    path = paths[0]  # (num_points, 4)
    return path  # numpy 로 쓰고 싶으면 path = np.array(path)


def main():
    # 1) PolySpline 기반 path 하나 샘플링
    path = sample_path_like_env()
    path = np.array(path)  # (num_points, 4)

    x_all = path[:, 0]
    y_all = path[:, 1]
    yaw_all = path[:, 2]
    lengths = path[:, 3]

    num_waypoints = 10
    std_waypoint_interval = 0.15
    interval = std_waypoint_interval * 2.0  # Isaac 코드에서 resample 시 쓰던 것과 비슷하게

    # 2) 길이에 따라 10개 waypoint index 선택 (PathCommand._initial_waypoints_indices 단일 env 버전)
    indices = select_waypoints_by_interval(lengths, num_waypoints, interval)
    wpts = path[indices, :]  # (10, 4)

    x_w = wpts[:, 0]
    y_w = wpts[:, 1]
    yaw_w = wpts[:, 2]

    # base frame 기준으로 보고 싶으면 root_yaw=0, root_pos=(0,0) 이라고 가정하면
    # world == base 와 동일해서 변환이 필요 없다.
    psi_orig = yaw_w.copy()

    # 3) 여러 g 값에 대해 yaw shaping
    g_list = [0.0, 0.5, 1.0]  # 원하는 값들로 바꿔서 테스트
    profiles = {}
    second_only = {}

    for g in g_list:
        yaw_blend, yaw_2nd = ayro_yaw_profile(psi_orig, g, blend=True)
        profiles[g] = yaw_blend
        second_only[g] = yaw_2nd

    # 4) 플롯
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    ax_xy = axes[0]
    ax_yaw = axes[1]

    # (a) XY 평면 궤적: 위치는 동일, heading 을 화살표로
    ax_xy.plot(x_w, y_w, "ko--", label="Waypoints (positions)")

    # 원래 yaw 프로파일 (검은 화살표)
    dx_orig = np.cos(psi_orig)
    dy_orig = np.sin(psi_orig)
    ax_xy.quiver(x_w, y_w, dx_orig, dy_orig, angles='xy', scale_units='xy',
                 scale=5.0, width=0.005, alpha=0.5, label="orig yaw")

    # 각 g 에 대해 yaw 화살표 추가
    colors = {0.0: "b", 0.5: "g", 1.0: "r"}
    for g in g_list:
        yaw_g = profiles[g]
        dx = np.cos(yaw_g)
        dy = np.sin(yaw_g)
        ax_xy.quiver(x_w, y_w, dx, dy, angles='xy', scale_units='xy',
                     scale=5.0, width=0.005, color=colors[g],
                     alpha=0.8, label=f"g={g:.2f}")

    ax_xy.set_aspect("equal", "box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Waypoint positions + heading (AYRO yaw shaping)")
    ax_xy.legend()

    # (b) Yaw vs waypoint index 플롯
    idxs = np.arange(num_waypoints)
    ax_yaw.plot(idxs, psi_orig, "k-o", label="orig yaw")

    for g in g_list:
        yaw_g = profiles[g]
        ax_yaw.plot(idxs, yaw_g, "-o", color=colors[g], label=f"g={g:.2f}")

    ax_yaw.set_xlabel("Waypoint index")
    ax_yaw.set_ylabel("Yaw [rad]")
    ax_yaw.set_title("Yaw profile per waypoint")
    ax_yaw.grid(True)
    ax_yaw.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
