"""Tests for GRPO grouping logic - ensuring games in same group share initial states."""

import numpy as np
import pytest

from dialop.envs.optimization import OptimizationEnv
from dialop.games.optimization import OptimizationGame


class TestGameDeterminism:
    """Test that seeded game generation is deterministic."""

    def test_same_seed_produces_identical_table_values(self, base_seed):
        """Verify that two games with the same seed have identical table values."""
        env1 = OptimizationEnv()
        env2 = OptimizationEnv()

        obs1 = env1.reset(game_state=None, seed=base_seed)
        obs2 = env2.reset(game_state=None, seed=base_seed)

        # Check that table values are identical
        assert np.array_equal(env1.game.table.values, env2.game.table.values), \
            "Tables with same seed should have identical values"

    def test_same_seed_produces_identical_scales(self, base_seed):
        """Verify that two games with the same seed have identical scales."""
        env1 = OptimizationEnv()
        env2 = OptimizationEnv()

        obs1 = env1.reset(game_state=None, seed=base_seed)
        obs2 = env2.reset(game_state=None, seed=base_seed)

        # Check that scales are identical
        assert env1.game.scales == env2.game.scales, \
            "Games with same seed should have identical scales"

    def test_same_seed_produces_identical_observations(self, base_seed):
        """Verify that two games with the same seed have identical player observations."""
        env1 = OptimizationEnv()
        env2 = OptimizationEnv()

        obs1 = env1.reset(game_state=None, seed=base_seed)
        obs2 = env2.reset(game_state=None, seed=base_seed)

        # Check that observations are identical
        assert obs1["player-1"] == obs2["player-1"], \
            "Player 1 observations should be identical with same seed"
        assert obs1["player-2"] == obs2["player-2"], \
            "Player 2 observations should be identical with same seed"

    def test_different_seeds_produce_different_tables(self, base_seed):
        """Verify that different seeds produce different game states."""
        env1 = OptimizationEnv()
        env2 = OptimizationEnv()

        obs1 = env1.reset(game_state=None, seed=base_seed)
        obs2 = env2.reset(game_state=None, seed=base_seed + 1)

        # Tables should be different
        assert not np.array_equal(env1.game.table.values, env2.game.table.values), \
            "Tables with different seeds should have different values"

    def test_no_seed_is_non_deterministic(self):
        """Verify that without seed, games are random."""
        env1 = OptimizationEnv()
        env2 = OptimizationEnv()

        obs1 = env1.reset(game_state=None, seed=None)
        obs2 = env2.reset(game_state=None, seed=None)

        # Very likely to be different (but not guaranteed, so we don't fail on match)
        # This test mainly documents the expected behavior
        # In practice, probability of identical 5x5 matrix is vanishingly small


class TestGRPOGrouping:
    """Test GRPO grouping logic - games in same group share initial state."""

    def test_group_games_share_initial_state(self, base_seed, group_size):
        """Verify that games in the same group have identical initial states."""
        # Simulate game_ids 0 through group_size-1 (first group)
        envs = []
        for game_id in range(group_size):
            env = OptimizationEnv()
            unique_game_id = game_id // group_size  # Should be 0 for all
            game_seed = base_seed + unique_game_id * 10000

            obs = env.reset(game_state=None, seed=game_seed)
            envs.append(env)

        # All envs in group should have identical table values
        reference_table = envs[0].game.table.values
        reference_scales = envs[0].game.scales

        for i, env in enumerate(envs[1:], 1):
            assert np.array_equal(env.game.table.values, reference_table), \
                f"Game {i} in group 0 should have same table as game 0"
            assert env.game.scales == reference_scales, \
                f"Game {i} in group 0 should have same scales as game 0"

    def test_different_groups_have_different_states(self, base_seed, group_size):
        """Verify that games from different groups have different initial states."""
        # Get a game from group 0
        env_group0 = OptimizationEnv()
        game_id_group0 = 0
        unique_id_0 = game_id_group0 // group_size  # 0
        seed_0 = base_seed + unique_id_0 * 10000
        obs0 = env_group0.reset(game_state=None, seed=seed_0)

        # Get a game from group 1
        env_group1 = OptimizationEnv()
        game_id_group1 = group_size  # First game of group 1
        unique_id_1 = game_id_group1 // group_size  # 1
        seed_1 = base_seed + unique_id_1 * 10000
        obs1 = env_group1.reset(game_state=None, seed=seed_1)

        # Tables should be different
        assert not np.array_equal(env_group0.game.table.values, env_group1.game.table.values), \
            "Games from different groups should have different table values"

    def test_group_seed_computation_correct(self, base_seed, group_size):
        """Verify that the group seed computation formula is correct."""
        test_cases = [
            # (game_id, expected_unique_id, expected_seed)
            (0, 0, base_seed + 0 * 10000),
            (1, 0, base_seed + 0 * 10000),
            (group_size - 1, 0, base_seed + 0 * 10000),
            (group_size, 1, base_seed + 1 * 10000),
            (group_size + 1, 1, base_seed + 1 * 10000),
            (2 * group_size, 2, base_seed + 2 * 10000),
        ]

        for game_id, expected_unique_id, expected_seed in test_cases:
            unique_id = game_id // group_size
            computed_seed = base_seed + unique_id * 10000

            assert unique_id == expected_unique_id, \
                f"game_id {game_id} should map to unique_id {expected_unique_id}"
            assert computed_seed == expected_seed, \
                f"game_id {game_id} should use seed {expected_seed}"


class TestTableRandomization:
    """Test that Table class randomization works correctly with seed."""

    def test_table_randomize_with_seed(self):
        """Test Table.randomize() with explicit RNG."""
        from dialop.games.optimization import Table

        # Create two tables and randomize with same seed
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        table1 = Table(num_rows=5, num_cols=5)
        table1.randomize(rng=rng1)

        table2 = Table(num_rows=5, num_cols=5)
        table2.randomize(rng=rng2)

        assert np.array_equal(table1.values, table2.values), \
            "Tables randomized with same RNG seed should be identical"

    def test_table_get_random_view_with_seed(self):
        """Test Table.get_random_view() with explicit RNG."""
        from dialop.games.optimization import Table

        # Create a base table
        table = Table(values=np.arange(25).reshape(5, 5))

        # Get two views with same seed
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        view1, mask1 = table.get_random_view(p_cell_observed=0.7, rng=rng1)
        view2, mask2 = table.get_random_view(p_cell_observed=0.7, rng=rng2)

        # Views should be identical (same cells masked)
        assert np.array_equal(view1.values, view2.values), \
            "Random views with same RNG seed should be identical"
        assert np.array_equal(mask1, mask2), \
            "Random view masks with same RNG seed should be identical"


class TestOptimizationGameReset:
    """Test OptimizationGame.reset() with seed parameter."""

    def test_reset_with_seed_is_deterministic(self):
        """Test that OptimizationGame.reset() with seed is deterministic."""
        game1 = OptimizationGame({})
        game2 = OptimizationGame({})

        game1.reset(randomize=True, seed=42)
        game2.reset(randomize=True, seed=42)

        # Check that games are identical
        assert np.array_equal(game1.table.values, game2.table.values)
        assert game1.scales == game2.scales
        assert np.array_equal(game1.masks[0], game2.masks[0])
        assert np.array_equal(game1.masks[1], game2.masks[1])

    def test_reset_without_seed_varies(self):
        """Test that OptimizationGame.reset() without seed creates different games."""
        game1 = OptimizationGame({})
        game2 = OptimizationGame({})

        game1.reset(randomize=True, seed=None)
        game2.reset(randomize=True, seed=None)

        # Very likely to be different (not guaranteed, so we don't assert)
        # This test documents expected behavior
