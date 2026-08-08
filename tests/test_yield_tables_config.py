"""
Tests for yield tables config-based data loading.

Tests that yield_tables.csv supports multiple configurations (central, alternative)
and that the DataHandler correctly filters and processes config-specific yield data.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from model.utils.data_handler import DataHandler


@pytest.fixture
def data_handler(test_data_dir):
    """Create DataHandler instance with test data."""
    return DataHandler(test_data_dir)


class TestYieldTablesConfigSupport:
    """Test that yield tables support config-based data selection."""

    def test_list_configs_includes_yield_tables(self, data_handler):
        """Verify that list_available_configs works for yield_tables component."""
        configs = data_handler.list_available_configs('yield_tables')
        assert isinstance(configs, list)
        assert len(configs) > 0
        assert 'central' in configs  # Test fixture should have central

    def test_get_yield_increments_defaults_to_central(self, data_handler):
        """Verify that get_yield_increments uses central config when not specified."""
        result = data_handler.get_yield_increments()

        # Should return dict with forest types
        assert isinstance(result, dict)
        assert 'permanent_exotic' in result
        assert 'production_exotic' in result
        assert 'natural_forest' in result

        # Each should be a numpy array
        for forest_type, increments in result.items():
            assert isinstance(increments, np.ndarray)
            assert len(increments) > 0
            assert np.all(increments >= 0)  # Increments should be non-negative

    def test_get_yield_increments_with_explicit_central(self, data_handler):
        """Verify that explicitly requesting central config works."""
        result_implicit = data_handler.get_yield_increments()
        result_explicit = data_handler.get_yield_increments(config='central')

        # Should produce identical results
        for forest_type in ['permanent_exotic', 'production_exotic', 'natural_forest']:
            np.testing.assert_array_equal(
                result_implicit[forest_type],
                result_explicit[forest_type]
            )

    def test_get_yield_increments_case_insensitive(self, data_handler):
        """Verify that config names are case-insensitive."""
        result_lower = data_handler.get_yield_increments(config='central')
        result_upper = data_handler.get_yield_increments(config='CENTRAL')
        result_mixed = data_handler.get_yield_increments(config='Central')

        # All should produce identical results
        for forest_type in ['permanent_exotic', 'production_exotic', 'natural_forest']:
            np.testing.assert_array_equal(
                result_lower[forest_type],
                result_upper[forest_type]
            )
            np.testing.assert_array_equal(
                result_lower[forest_type],
                result_mixed[forest_type]
            )

    def test_get_yield_increments_invalid_config_raises_error(self, data_handler):
        """Verify that requesting non-existent config raises clear error."""
        with pytest.raises(KeyError) as exc_info:
            data_handler.get_yield_increments(config='nonexistent')

        error_message = str(exc_info.value)
        assert 'nonexistent' in error_message
        assert 'Available configs' in error_message

    def test_yield_increment_calculation_properties(self, data_handler):
        """Verify that yield increments maintain expected mathematical properties."""
        result = data_handler.get_yield_increments(config='central')

        for forest_type, increments in result.items():
            # First increment should equal first cumulative value
            assert increments[0] > 0

            # All increments should be non-negative (yield doesn't decrease)
            assert np.all(increments >= 0)

            # Cumulative sum should be monotonically increasing
            cumulative = np.cumsum(increments)
            assert np.all(np.diff(cumulative) >= 0)

    def test_production_exotic_truncated_at_average_age(self, data_handler):
        """Verify that production_exotic yields are truncated at average_age."""
        result = data_handler.get_yield_increments(config='central')

        # Get average_age from manley parameters (default 16)
        manley_params = data_handler.get_manley_parameters(config='central')
        average_age = int(manley_params.get('average_age', 16))

        # Production exotic should have exactly average_age increments
        assert len(result['production_exotic']) == average_age

        # Permanent and natural forest should have more
        assert len(result['permanent_exotic']) > average_age
        assert len(result['natural_forest']) > average_age

    def test_blend_weights_applied_correctly(self, data_handler):
        """Verify that large/small forest blending uses correct weights."""
        # Get the raw data to manually compute expected blend
        yield_data = data_handler.yield_tables_data
        config_data = yield_data[yield_data['Config'].str.lower() == 'central']

        manley_params = data_handler.get_manley_parameters(config='central')
        large_weight = manley_params.get('large_forest_weight', 0.8)
        small_weight = manley_params.get('small_forest_weight', 0.2)

        # Test for permanent_exotic
        large_data = config_data[
            (config_data['Forest'] == 'permanent_exotic') &
            (config_data['Size'] == 'large')
        ][['Age', 'Value']].sort_values('Age')

        small_data = config_data[
            (config_data['Forest'] == 'permanent_exotic') &
            (config_data['Size'] == 'small')
        ][['Age', 'Value']].sort_values('Age')

        # Manually compute blended cumulative
        cumulative_large = large_data['Value'].values.astype(float)
        cumulative_small = small_data['Value'].values.astype(float)
        expected_blended = large_weight * cumulative_large + small_weight * cumulative_small
        expected_increments = np.diff(expected_blended, prepend=0)

        # Get actual result
        result = data_handler.get_yield_increments(config='central')

        # Should match within floating point precision
        np.testing.assert_allclose(
            result['permanent_exotic'],
            expected_increments,
            rtol=1e-10
        )


class TestYieldTablesIntegrationWithProduction:
    """Test yield tables work with actual production data (if available)."""

    def test_production_data_has_multiple_configs(self):
        """Verify that production yield_tables.csv has both central and alternative configs."""
        production_path = Path('data/inputs/forestry/yield_tables.csv')

        if not production_path.exists():
            pytest.skip("Production yield_tables.csv not found")

        df = pd.read_csv(production_path)

        # Should have Config column
        assert 'Config' in df.columns

        # Should have at least central config
        configs = df['Config'].unique()
        assert 'central' in configs or 'Central' in configs

        # Should have the expected columns
        assert 'Age' in df.columns
        assert 'Forest' in df.columns
        assert 'Size' in df.columns
        assert 'Value' in df.columns

    def test_production_central_matches_original_data(self):
        """Verify that central config in new format matches original data structure."""
        new_path = Path('data/inputs/forestry/yield_tables.csv')
        original_path = Path('data/inputs/forestry/yield_tables_original.csv')

        if not new_path.exists() or not original_path.exists():
            pytest.skip("Production yield tables not found")

        # Load both
        new_df = pd.read_csv(new_path)
        original_df = pd.read_csv(original_path)

        # Get central config from new format
        central_data = new_df[new_df['Config'].str.lower() == 'central']

        # Verify we can reconstruct original structure
        for forest in ['permanent_exotic', 'production_exotic', 'natural_forest']:
            for size in ['large', 'small']:
                col_name = f"{forest}_{size}"

                # Get data from new format
                new_values = central_data[
                    (central_data['Forest'] == forest) &
                    (central_data['Size'] == size)
                ].sort_values('Age')['Value'].values

                # Get data from original format
                if col_name in original_df.columns:
                    original_values = original_df[col_name].dropna().values

                    # Should have same length and values
                    assert len(new_values) == len(original_values)
                    np.testing.assert_allclose(new_values, original_values, rtol=1e-10)

    def test_alternative_config_loads_successfully(self):
        """Verify that alternative config (new MPI data) loads if present."""
        production_path = Path('data/inputs/forestry/yield_tables.csv')

        if not production_path.exists():
            pytest.skip("Production yield_tables.csv not found")

        df = pd.read_csv(production_path)
        configs = df['Config'].unique()

        if 'alternative' not in [c.lower() for c in configs]:
            pytest.skip("Alternative config not present in production data")

        # Create data handler and load alternative config
        data_handler = DataHandler(Path('data'))
        result = data_handler.get_yield_increments(config='alternative')

        # Should successfully return data
        assert isinstance(result, dict)
        assert len(result) == 3
        for forest_type, increments in result.items():
            assert isinstance(increments, np.ndarray)
            assert len(increments) > 0
