"""Smoke tests for chart generation — verify charts render without error."""

import pytest
from model.interface.chart_generator import ChartGenerator


class TestSingleCharts:
    def test_carbon_price_chart(self, run_model):
        charts = ChartGenerator(run_model)
        fig = charts.carbon_price_chart()
        assert fig is not None

    def test_emissions_pathway_chart(self, run_model):
        charts = ChartGenerator(run_model)
        fig = charts.emissions_pathway_chart()
        assert fig is not None

    def test_supply_components_chart(self, run_model):
        charts = ChartGenerator(run_model)
        fig = charts.supply_components_chart()
        assert fig is not None

    def test_stockpile_balance_chart(self, run_model):
        charts = ChartGenerator(run_model)
        fig = charts.stockpile_balance_chart()
        assert fig is not None

    def test_supply_demand_balance_chart(self, run_model):
        charts = ChartGenerator(run_model)
        fig = charts.supply_demand_balance_chart()
        assert fig is not None

    def test_auction_volume_revenue_chart(self, run_model):
        charts = ChartGenerator(run_model)
        fig = charts.auction_volume_revenue_chart()
        assert fig is not None

    def test_generate_standard_charts(self, run_model):
        charts = ChartGenerator(run_model)
        result = charts.generate_standard_charts()
        assert isinstance(result, dict)
        assert len(result) > 0
        for name, fig in result.items():
            assert fig is not None, f"Chart '{name}' is None"
