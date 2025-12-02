# Test Outputs Directory

This directory contains visualization outputs and test results from the star detection system.

## Contents

- **Visualization images**: PNG files showing detected stars overlaid on input images
- **Test results**: Output files from running detection tests

## Files

The following test output files are currently stored here (and ignored by git):

- `demo_output.png` - Default demo output
- `test_*.png` - Various test visualization outputs
- `result.png` - Custom test result
- `final_test*.png` - Final verification test outputs

## Regenerating Outputs

To regenerate these outputs, run:

```bash
python demo_detection.py --visualize --save-vis test_outputs/your_output.png
```

Or use the demo script with other parameters to generate new test visualizations.

