# Test Fixtures

These CSV files contain synthetic input data for NZUpy's test suite.
They mirror the structure of the real input files in `data/inputs/` but use
simplified, round numbers designed for easy manual verification.

**Do NOT update these files when real input data changes.**
That's the whole point — tests use frozen fixtures so they are immune to
data updates in `data/inputs/`.

If the model's CSV format changes (new columns, renamed fields), update
these fixtures to match the new format.
