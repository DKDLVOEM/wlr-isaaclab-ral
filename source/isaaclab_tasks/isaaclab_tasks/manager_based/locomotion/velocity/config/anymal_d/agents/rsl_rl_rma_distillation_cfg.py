# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class AnymalDRoughRMADistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Student distillation with history observations for RMA."""

    num_steps_per_env = 120
    max_iterations = 300
    save_interval = 50
    experiment_name = "anymal_d_rough_rma_student"
    obs_groups = {"policy": ["policy", "history"], "teacher": ["policy", "privileged"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_hidden_dims=[128, 128, 128],
        teacher_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


@configclass
class AnymalDFlatRMADistillationRunnerCfg(AnymalDRoughRMADistillationRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat_rma_student"
