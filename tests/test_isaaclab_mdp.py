"""Comprehensive unit tests for the Isaac Lab MDP pure-Python layer.

Tests cover:
    - task_geometry: zone centres, square centres, piece positions, distances.
    - trial_state: TrialStateManager phase progression, reset, step counters.
    - mdp_logic: observation, reward, check, and batch kernels.

No Isaac Lab, no torch, no GPU required — all pure Python.
"""

from __future__ import annotations

import pytest

from isaac.isaaclab_tasks.synria_pickplace.mdp_logic import (
    batch_ee_to_target_distance,
    batch_piece_dropped,
    batch_piece_in_zone,
    batch_piece_returned,
    check_grasp_criteria,
    check_piece_dropped,
    check_piece_in_zone,
    check_piece_returned,
    check_task_success_v1,
    obs_ee_pose_flat,
    obs_joint_state_flat,
    obs_target_piece_onehot,
    obs_target_zone_onehot,
    rew_action_norm,
    rew_arm_collision,
    rew_ee_to_target_distance,
    rew_grasp_confirmed,
    rew_gripper_piece_penalty,
    rew_piece_dropped,
    rew_piece_in_zone,
    rew_piece_returned_to_origin,
)

# ---------------------------------------------------------------------------
# Module imports — these must work on any machine
# ---------------------------------------------------------------------------
from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
    BOARD_CENTER_X,
    BOARD_CENTER_Y,
    BOARD_SIZES,
    GRASP_APPROACH_RADIUS_M,
    GRASP_CONFIRM_STEPS,
    GRIPPER_GRASP_WIDTH_M,
    GRIPPER_OPEN_M,
    MAX_PIECES,
    PIECE_COUNTS,
    PIECE_DROP_THRESHOLD_M,
    PIECE_IN_ZONE_TOLERANCE_M,
    PIECE_RETURNED_TOLERANCE_M,
    STAGING_ZONE_DEPTH,
    STAGING_ZONE_OFFSET,
    TABLE_SURFACE_Z,
    ZONE_NAMES,
    l2_distance_2d,
    l2_distance_3d,
    piece_start_xy,
    square_center_xy,
    zone_center_xy,
)
from isaac.isaaclab_tasks.synria_pickplace.trial_state import (
    TaskPhase,
    TrialStateManager,
)

# ===========================================================================
# task_geometry tests
# ===========================================================================


class TestConstants:
    def test_table_surface_z(self) -> None:
        # TABLE_HEIGHT(0.75) + TABLE_THICKNESS(0.04)
        assert abs(TABLE_SURFACE_Z - 0.79) < 1e-9

    def test_max_pieces_is_32(self) -> None:
        assert MAX_PIECES == 32

    def test_piece_counts(self) -> None:
        assert PIECE_COUNTS["chess"] == 32
        assert PIECE_COUNTS["checkers"] == 24
        assert PIECE_COUNTS["ludo"] == 16

    def test_board_sizes_equal(self) -> None:
        # All three games share the same 0.44 m board
        for game in ("chess", "checkers", "ludo"):
            assert BOARD_SIZES[game] == 0.44

    def test_zone_names_order(self) -> None:
        assert ZONE_NAMES == ("left", "right", "top", "bottom")

    def test_gripper_constants(self) -> None:
        # Prismatic fingers: 0.025 m travel each → 0.05 m max width. The
        # grasp threshold must exceed the 0.04 m piece diameter (fingers
        # stall at the piece), else grasp confirmation can never fire.
        assert GRIPPER_OPEN_M == 0.05
        assert GRIPPER_GRASP_WIDTH_M == 0.045
        # 1 step: the lift requirement makes a single step of
        # holding-band + proximity + airborne unfakeable evidence.
        assert GRASP_CONFIRM_STEPS == 1


class TestZoneCenterXY:
    """Zone centre = board_centre ± (half_board + STAGING_ZONE_OFFSET + STAGING_ZONE_DEPTH/2)."""

    def _expected_clearance(self, game: str) -> float:
        return BOARD_SIZES[game] / 2 + STAGING_ZONE_OFFSET + STAGING_ZONE_DEPTH / 2

    def test_left_zone_chess(self) -> None:
        x, y = zone_center_xy("left", "chess")
        c = self._expected_clearance("chess")
        assert abs(x - BOARD_CENTER_X) < 1e-9
        assert abs(y - (BOARD_CENTER_Y - c)) < 1e-9

    def test_right_zone_chess(self) -> None:
        x, y = zone_center_xy("right", "chess")
        c = self._expected_clearance("chess")
        assert abs(x - BOARD_CENTER_X) < 1e-9
        assert abs(y - (BOARD_CENTER_Y + c)) < 1e-9

    def test_top_zone_chess(self) -> None:
        x, y = zone_center_xy("top", "chess")
        c = self._expected_clearance("chess")
        assert abs(x - (BOARD_CENTER_X - c)) < 1e-9
        assert abs(y - BOARD_CENTER_Y) < 1e-9

    def test_bottom_zone_chess(self) -> None:
        x, y = zone_center_xy("bottom", "chess")
        c = self._expected_clearance("chess")
        assert abs(x - (BOARD_CENTER_X + c)) < 1e-9
        assert abs(y - BOARD_CENTER_Y) < 1e-9

    def test_left_right_symmetric_y(self) -> None:
        _, yl = zone_center_xy("left", "chess")
        _, yr = zone_center_xy("right", "chess")
        assert abs(yl + yr - 2 * BOARD_CENTER_Y) < 1e-9  # symmetric about board centre

    def test_top_bottom_symmetric_x(self) -> None:
        xt, _ = zone_center_xy("top", "chess")
        xb, _ = zone_center_xy("bottom", "chess")
        assert abs(xt + xb - 2 * BOARD_CENTER_X) < 1e-9

    def test_invalid_zone_raises(self) -> None:
        with pytest.raises(ValueError, match="zone"):
            zone_center_xy("diagonal", "chess")

    def test_invalid_game_raises(self) -> None:
        with pytest.raises(ValueError, match="game"):
            zone_center_xy("left", "go")

    def test_all_games_return_float_pairs(self) -> None:
        for game in ("chess", "checkers", "ludo"):
            for zone in ZONE_NAMES:
                xy = zone_center_xy(zone, game)
                assert len(xy) == 2
                assert all(isinstance(v, float) for v in xy)


class TestSquareCenterXY:
    def test_chess_origin_corner(self) -> None:
        # Row 0, col 0 → closest to arm and operator's left
        x, y = square_center_xy("chess", 0, 0)
        sq = BOARD_SIZES["chess"] / 8
        expected_x = BOARD_CENTER_X - BOARD_SIZES["chess"] / 2 + sq * 0.5
        expected_y = BOARD_CENTER_Y - BOARD_SIZES["chess"] / 2 + sq * 0.5
        assert abs(x - expected_x) < 1e-4
        assert abs(y - expected_y) < 1e-4

    def test_chess_centre_square_near_board_centre(self) -> None:
        # row=3, col=3 and row=4, col=4 should straddle board centre
        x3, y3 = square_center_xy("chess", 3, 3)
        x4, y4 = square_center_xy("chess", 4, 4)
        mid_x = (x3 + x4) / 2
        mid_y = (y3 + y4) / 2
        assert abs(mid_x - BOARD_CENTER_X) < 1e-4
        assert abs(mid_y - BOARD_CENTER_Y) < 1e-4

    def test_square_spacing_chess(self) -> None:
        sq = BOARD_SIZES["chess"] / 8
        x0, _ = square_center_xy("chess", 0, 0)
        x1, _ = square_center_xy("chess", 1, 0)
        assert abs((x1 - x0) - sq) < 1e-5


class TestPieceStartXY:
    def test_chess_piece_0_row0_col0(self) -> None:
        # Piece 0 = white rook at (row=0, col=0)
        xy = piece_start_xy("chess", 0)
        expected = square_center_xy("chess", 0, 0)
        assert abs(xy[0] - expected[0]) < 1e-9
        assert abs(xy[1] - expected[1]) < 1e-9

    def test_chess_all_pieces_in_range(self) -> None:
        for idx in range(32):
            x, y = piece_start_xy("chess", idx)
            assert -1.0 < x < 1.0
            assert -1.0 < y < 1.0

    def test_ludo_piece_count(self) -> None:
        for idx in range(16):
            piece_start_xy("ludo", idx)  # must not raise

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(IndexError):
            piece_start_xy("chess", 100)

    def test_checkers_dark_squares_only(self) -> None:
        # All 24 checkers pieces should be on dark squares (row+col odd)
        from isaac.isaaclab_tasks.synria_pickplace.task_geometry import INITIAL_POSITIONS

        for row, col in INITIAL_POSITIONS["checkers"]:
            assert (row + col) % 2 == 1, f"checker on light square ({row},{col})"


class TestDistanceHelpers:
    def test_l2_2d_zero(self) -> None:
        assert l2_distance_2d((0.0, 0.0), (0.0, 0.0)) == 0.0

    def test_l2_2d_unit(self) -> None:
        assert abs(l2_distance_2d((0.0, 0.0), (1.0, 0.0)) - 1.0) < 1e-9

    def test_l2_2d_pythagorean(self) -> None:
        assert abs(l2_distance_2d((0.0, 0.0), (3.0, 4.0)) - 5.0) < 1e-9

    def test_l2_3d_unit(self) -> None:
        assert abs(l2_distance_3d((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) - 1.0) < 1e-9

    def test_l2_3d_pythagorean(self) -> None:
        # 1² + 2² + 2² = 9 → dist = 3
        assert abs(l2_distance_3d((0.0, 0.0, 0.0), (1.0, 2.0, 2.0)) - 3.0) < 1e-9


# ===========================================================================
# trial_state tests
# ===========================================================================


class TestTaskPhase:
    def test_phase_ordering(self) -> None:
        assert TaskPhase.PICK_FROM_BOARD < TaskPhase.PLACE_ON_ZONE
        assert TaskPhase.PLACE_ON_ZONE < TaskPhase.PICK_FROM_ZONE
        assert TaskPhase.PICK_FROM_ZONE < TaskPhase.RETURN_TO_BOARD
        assert TaskPhase.RETURN_TO_BOARD < TaskPhase.SUCCESS
        assert TaskPhase.SUCCESS < TaskPhase.FAILED

    def test_phase_values(self) -> None:
        assert int(TaskPhase.PICK_FROM_BOARD) == 0
        assert int(TaskPhase.FAILED) == 5


class TestTrialStateManager:
    def _mgr(self, num_envs: int = 4, game: str = "chess", seed: int = 42) -> TrialStateManager:
        return TrialStateManager(num_envs=num_envs, game=game, seed=seed)

    def test_init_all_pick_from_board(self) -> None:
        mgr = self._mgr()
        for i in range(4):
            assert mgr.get_phase(i) == TaskPhase.PICK_FROM_BOARD

    def test_init_valid_piece_indices(self) -> None:
        mgr = self._mgr(num_envs=16, game="chess")
        for i in range(16):
            idx = mgr.get_state(i).target_piece_idx
            assert 0 <= idx < PIECE_COUNTS["chess"]

    def test_init_valid_zone_indices(self) -> None:
        mgr = self._mgr(num_envs=16)
        for i in range(16):
            idx = mgr.get_state(i).target_zone_idx
            assert 0 <= idx < 4

    def test_invalid_game_raises(self) -> None:
        with pytest.raises(ValueError, match="game"):
            TrialStateManager(num_envs=1, game="monopoly")

    def test_invalid_num_envs_raises(self) -> None:
        with pytest.raises(ValueError, match="num_envs"):
            TrialStateManager(num_envs=0, game="chess")

    def test_advance_phase_sequence(self) -> None:
        mgr = self._mgr(num_envs=1)
        expected = [
            TaskPhase.PICK_FROM_BOARD,
            TaskPhase.PLACE_ON_ZONE,
            TaskPhase.PICK_FROM_ZONE,
            TaskPhase.RETURN_TO_BOARD,
            TaskPhase.SUCCESS,
            TaskPhase.SUCCESS,  # clamped at SUCCESS
        ]
        for exp in expected:
            assert mgr.get_phase(0) == exp
            mgr.advance_phase(0)

    def test_mark_failed(self) -> None:
        mgr = self._mgr(num_envs=1)
        mgr.mark_failed(0)
        assert mgr.get_phase(0) == TaskPhase.FAILED
        assert mgr.is_failed(0)
        assert not mgr.is_success(0)

    def test_is_success(self) -> None:
        mgr = self._mgr(num_envs=1)
        for _ in range(4):
            mgr.advance_phase(0)
        assert mgr.is_success(0)

    def test_step_increments_counters(self) -> None:
        mgr = self._mgr(num_envs=1)
        for _ in range(5):
            mgr.step(0)
        s = mgr.get_state(0)
        assert s.episode_step == 5
        assert s.phase_step_count == 5

    def test_advance_phase_resets_phase_step_count(self) -> None:
        mgr = self._mgr(num_envs=1)
        mgr.step(0)
        mgr.step(0)
        mgr.advance_phase(0)
        assert mgr.get_state(0).phase_step_count == 0
        assert mgr.get_state(0).episode_step == 2  # episode steps preserved

    def test_grasp_counter_increment_and_reset(self) -> None:
        mgr = self._mgr(num_envs=1)
        mgr.increment_grasp(0)
        mgr.increment_grasp(0)
        assert mgr.get_state(0).consec_grasp_steps == 2
        mgr.reset_grasp(0)
        assert mgr.get_state(0).consec_grasp_steps == 0

    def test_reset_env_ids(self) -> None:
        mgr = self._mgr(num_envs=4)
        # Advance envs 0 and 2
        mgr.advance_phase(0)
        mgr.advance_phase(2)
        # Reset only env 0
        mgr.reset([0])
        assert mgr.get_phase(0) == TaskPhase.PICK_FROM_BOARD
        assert mgr.get_phase(2) == TaskPhase.PLACE_ON_ZONE  # unchanged

    def test_reset_all_envs(self) -> None:
        mgr = self._mgr(num_envs=4)
        for i in range(4):
            mgr.advance_phase(i)
        mgr.reset()  # default = all envs
        for i in range(4):
            assert mgr.get_phase(i) == TaskPhase.PICK_FROM_BOARD

    def test_current_target_xy_pick_from_board(self) -> None:
        mgr = self._mgr(num_envs=1)
        # Phase 0: target = origin_xy
        state = mgr.get_state(0)
        assert mgr.current_target_xy(0) == state.origin_xy

    def test_current_target_xy_place_on_zone(self) -> None:
        mgr = self._mgr(num_envs=1)
        mgr.advance_phase(0)  # → PLACE_ON_ZONE
        state = mgr.get_state(0)
        assert mgr.current_target_xy(0) == state.zone_xy

    def test_current_target_xy_pick_from_zone(self) -> None:
        mgr = self._mgr(num_envs=1)
        mgr.advance_phase(0)
        mgr.advance_phase(0)  # → PICK_FROM_ZONE
        state = mgr.get_state(0)
        assert mgr.current_target_xy(0) == state.zone_xy

    def test_current_target_xy_return_to_board(self) -> None:
        mgr = self._mgr(num_envs=1)
        for _ in range(3):
            mgr.advance_phase(0)  # → RETURN_TO_BOARD
        state = mgr.get_state(0)
        assert mgr.current_target_xy(0) == state.origin_xy

    def test_all_target_xys_length(self) -> None:
        mgr = self._mgr(num_envs=8)
        xys = mgr.all_target_xys()
        assert len(xys) == 8

    def test_get_phases_returns_all(self) -> None:
        mgr = self._mgr(num_envs=3)
        phases = mgr.get_phases()
        assert len(phases) == 3
        assert all(p == TaskPhase.PICK_FROM_BOARD for p in phases)

    def test_num_envs_property(self) -> None:
        assert self._mgr(num_envs=7).num_envs == 7

    def test_game_property(self) -> None:
        assert self._mgr(game="ludo").game == "ludo"

    def test_checkers_piece_range(self) -> None:
        mgr = TrialStateManager(num_envs=50, game="checkers", seed=0)
        for i in range(50):
            idx = mgr.get_state(i).target_piece_idx
            assert 0 <= idx < PIECE_COUNTS["checkers"]


# ===========================================================================
# mdp_logic tests
# ===========================================================================


class TestObsKernels:
    def test_obs_joint_state_flat_length(self) -> None:
        v = obs_joint_state_flat([0.1] * 6, [0.2] * 6, 0.05)
        assert len(v) == 13

    def test_obs_joint_state_flat_order(self) -> None:
        pos = [float(i) for i in range(6)]
        vel = [float(i + 10) for i in range(6)]
        v = obs_joint_state_flat(pos, vel, 0.042)
        assert v[:6] == pos
        assert v[6:12] == vel
        assert v[12] == 0.042

    def test_obs_target_piece_onehot_structure(self) -> None:
        for idx in range(MAX_PIECES):
            oh = obs_target_piece_onehot(idx)
            assert len(oh) == MAX_PIECES
            assert oh[idx] == 1.0
            assert sum(oh) == 1.0

    def test_obs_target_piece_onehot_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            obs_target_piece_onehot(-1)
        with pytest.raises(ValueError):
            obs_target_piece_onehot(MAX_PIECES)

    def test_obs_target_zone_onehot_structure(self) -> None:
        for idx in range(4):
            oh = obs_target_zone_onehot(idx)
            assert len(oh) == 4
            assert oh[idx] == 1.0
            assert sum(oh) == 1.0

    def test_obs_target_zone_onehot_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            obs_target_zone_onehot(4)

    def test_obs_ee_pose_flat_default_quat(self) -> None:
        v = obs_ee_pose_flat((0.1, 0.2, 0.5))
        assert v == [0.1, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0]

    def test_obs_ee_pose_flat_custom_quat(self) -> None:
        v = obs_ee_pose_flat((0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
        assert v == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


class TestRewardKernels:
    def test_rew_ee_to_target_zero_distance(self) -> None:
        assert rew_ee_to_target_distance((0.3, 0.1), (0.3, 0.1)) == 0.0

    def test_rew_ee_to_target_negative(self) -> None:
        r = rew_ee_to_target_distance((0.0, 0.0), (0.3, 0.4))
        assert abs(r - (-0.5)) < 1e-9  # 3-4-5 triangle, dist=0.5

    def test_rew_gripper_piece_penalty_open_gripper(self) -> None:
        # Gripper open → no penalty regardless of distance
        r = rew_gripper_piece_penalty(
            gripper_xy=(0.0, 0.0),
            piece_xy=(1.0, 1.0),
            gripper_open_m=GRIPPER_GRASP_WIDTH_M * 3.0,  # well above threshold
        )
        assert r == 0.0

    def test_rew_gripper_piece_penalty_closed_gripper(self) -> None:
        r = rew_gripper_piece_penalty(
            gripper_xy=(0.0, 0.0),
            piece_xy=(0.0, 0.01),
            gripper_open_m=GRIPPER_GRASP_WIDTH_M * 0.5,  # well below threshold
        )
        assert r < 0.0
        assert abs(r - (-0.01)) < 1e-9

    def test_rew_action_norm_zero(self) -> None:
        assert rew_action_norm([0.0] * 7) == 0.0

    def test_rew_action_norm_unit_vector(self) -> None:
        v = [1.0] + [0.0] * 6
        assert abs(rew_action_norm(v) - 1.0) < 1e-9

    def test_rew_grasp_confirmed_wrong_count(self) -> None:
        assert rew_grasp_confirmed(GRASP_CONFIRM_STEPS - 1) == 0.0
        assert rew_grasp_confirmed(GRASP_CONFIRM_STEPS + 1) == 0.0

    def test_rew_grasp_confirmed_exact_count(self) -> None:
        assert rew_grasp_confirmed(GRASP_CONFIRM_STEPS) == 1.0

    def test_rew_piece_in_zone_inside(self) -> None:
        tol = PIECE_IN_ZONE_TOLERANCE_M
        r = rew_piece_in_zone(
            piece_xy=(0.3, -0.3),
            zone_xy=(0.3, -0.3 + tol * 0.5),
        )
        assert r == 1.0

    def test_rew_piece_in_zone_outside(self) -> None:
        tol = PIECE_IN_ZONE_TOLERANCE_M
        r = rew_piece_in_zone(
            piece_xy=(0.3, -0.3),
            zone_xy=(0.3, -0.3 + tol * 2.0),
        )
        assert r == 0.0

    def test_rew_piece_returned_inside(self) -> None:
        tol = PIECE_RETURNED_TOLERANCE_M
        r = rew_piece_returned_to_origin(
            piece_xy=(0.1, 0.1),
            origin_xy=(0.1 + tol * 0.5, 0.1),
        )
        assert r == 1.0

    def test_rew_piece_returned_outside(self) -> None:
        tol = PIECE_RETURNED_TOLERANCE_M
        r = rew_piece_returned_to_origin(
            piece_xy=(0.1, 0.1),
            origin_xy=(0.1 + tol * 2.0, 0.1),
        )
        assert r == 0.0

    def test_rew_piece_dropped_not_dropped(self) -> None:
        # Piece on table surface
        assert rew_piece_dropped(TABLE_SURFACE_Z) == 0.0

    def test_rew_piece_dropped_below(self) -> None:
        z = TABLE_SURFACE_Z - PIECE_DROP_THRESHOLD_M - 0.001
        assert rew_piece_dropped(z) == -1.0

    def test_rew_arm_collision_false(self) -> None:
        assert rew_arm_collision(False) == 0.0

    def test_rew_arm_collision_true(self) -> None:
        assert rew_arm_collision(True) == -1.0


class TestCheckKernels:
    def test_check_grasp_criteria_both_met(self) -> None:
        assert check_grasp_criteria(
            gripper_open_m=GRIPPER_GRASP_WIDTH_M * 0.9,
            gripper_to_piece_dist_m=GRASP_APPROACH_RADIUS_M * 0.9,
        )

    def test_check_grasp_criteria_gripper_open(self) -> None:
        assert not check_grasp_criteria(
            gripper_open_m=GRIPPER_GRASP_WIDTH_M * 1.1,
            gripper_to_piece_dist_m=GRASP_APPROACH_RADIUS_M * 0.5,
        )

    def test_check_grasp_criteria_piece_too_far(self) -> None:
        assert not check_grasp_criteria(
            gripper_open_m=GRIPPER_GRASP_WIDTH_M * 0.5,
            gripper_to_piece_dist_m=GRASP_APPROACH_RADIUS_M * 1.1,
        )

    def test_check_piece_in_zone_edge_case(self) -> None:
        tol = PIECE_IN_ZONE_TOLERANCE_M
        zone = (0.3, -0.3)
        # Slightly inside tolerance boundary (avoiding fp precision issues at exact boundary)
        piece = (0.3, -0.3 + tol * 0.999)
        assert check_piece_in_zone(piece, zone)

    def test_check_piece_in_zone_just_outside(self) -> None:
        tol = PIECE_IN_ZONE_TOLERANCE_M
        zone = (0.3, -0.3)
        piece = (0.3, -0.3 + tol + 1e-6)
        assert not check_piece_in_zone(piece, zone)

    def test_check_piece_returned(self) -> None:
        tol = PIECE_RETURNED_TOLERANCE_M
        origin = (0.1, 0.2)
        assert check_piece_returned((0.1 + tol * 0.9, 0.2), origin)
        assert not check_piece_returned((0.1 + tol * 1.1, 0.2), origin)

    def test_check_piece_dropped_false(self) -> None:
        z = TABLE_SURFACE_Z - PIECE_DROP_THRESHOLD_M + 0.001
        assert not check_piece_dropped(z)

    def test_check_piece_dropped_true(self) -> None:
        z = TABLE_SURFACE_Z - PIECE_DROP_THRESHOLD_M - 0.001
        assert check_piece_dropped(z)

    def test_check_task_success_v1_wrong_phase(self) -> None:
        # Not in RETURN_TO_BOARD phase → never success
        piece_xy = (0.1, 0.1)
        origin_xy = (0.1, 0.1)
        assert not check_task_success_v1(
            int(TaskPhase.PLACE_ON_ZONE), piece_xy, origin_xy
        )

    def test_check_task_success_v1_correct(self) -> None:
        piece_xy = (0.1, 0.1)
        origin_xy = (0.1, 0.1)
        assert check_task_success_v1(
            int(TaskPhase.RETURN_TO_BOARD), piece_xy, origin_xy
        )

    def test_check_task_success_v1_piece_out_of_tolerance(self) -> None:
        piece_xy = (0.5, 0.5)
        origin_xy = (0.1, 0.1)
        assert not check_task_success_v1(
            int(TaskPhase.RETURN_TO_BOARD), piece_xy, origin_xy
        )


class TestBatchHelpers:
    def test_batch_ee_to_target_distance_length(self) -> None:
        ee_xys = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)]
        target_xys = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)]
        result = batch_ee_to_target_distance(ee_xys, target_xys)
        assert len(result) == 3
        assert all(r == 0.0 for r in result)

    def test_batch_ee_to_target_distance_values(self) -> None:
        ee_xys = [(0.0, 0.0)]
        target_xys = [(0.3, 0.4)]
        result = batch_ee_to_target_distance(ee_xys, target_xys)
        assert abs(result[0] - (-0.5)) < 1e-9

    def test_batch_piece_in_zone_all_inside(self) -> None:
        piece_xys = [(0.3, -0.3), (0.1, 0.1)]
        zone_xys = [(0.3, -0.3), (0.1, 0.1)]
        result = batch_piece_in_zone(piece_xys, zone_xys)
        assert all(r == 1.0 for r in result)

    def test_batch_piece_returned(self) -> None:
        piece_xys = [(0.1, 0.1)]
        origin_xys = [(0.1, 0.1)]
        result = batch_piece_returned(piece_xys, origin_xys)
        assert result[0] == 1.0

    def test_batch_piece_dropped_mixed(self) -> None:
        z_above = TABLE_SURFACE_Z
        z_below = TABLE_SURFACE_Z - PIECE_DROP_THRESHOLD_M - 0.01
        result = batch_piece_dropped([z_above, z_below])
        assert result[0] == 0.0
        assert result[1] == -1.0

    def test_batch_lengths_match(self) -> None:
        n = 10
        piece_xys = [(0.0, 0.0)] * n
        zone_xys = [(0.1, 0.1)] * n
        assert len(batch_piece_in_zone(piece_xys, zone_xys)) == n
        assert len(batch_piece_returned(piece_xys, zone_xys)) == n
        assert len(batch_piece_dropped([0.5] * n)) == n
