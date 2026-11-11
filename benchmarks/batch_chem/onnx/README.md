# Batch Chemistry ONNX Benchmarks

This directory contains batch chemistry input files that exercise the ONNX chemistry engine with LSURF models and EX8 neural-network and random-forest models. Each case uses `hands_off = true`, the ONNX engine, the `initial` condition from the referenced model config JSON file, verbose gnuplot output, and these shared state/material settings:

- `timestep = 864000`
- `density = 997.16`
- `porosity = 0.25`
- `temperature = 25.0`
- `pressure = 101325.0`
- `volume = 1.0`
- `saturation = 1.0`

Run a case from this directory with the installed batch chemistry driver, for example:

```bash
cd [project_root]/benchmarks/batch_chem/onnx
../../../build/install/bin/batch_chem ex8-rf-9-dynamic-batch.cfg
```

## Baseline Input Conditions

> **Note on Units:** The values defined in the JSON `conditions` block follow **Alquimia's standard unit conventions**. However, during mobile and immobile model inference, the actual input and output tensors use **[molarity]**. The Alquimia interface handles the necessary conversions between the Alquimia state and the model tensors.

The baseline values below come from the `conditions` block in each referenced model config JSON file:

```json
"conditions": {
  "initial": {
    ...
  }
}
```

The model config maps those named values into specific `AlquimiaState` fields and ONNX tensor elements.

### EX8 Integrated Neural-Network Baseline

Used by:

- `ex8-nn-1d.cfg`
- `ex8-nn-batch1.cfg`
- `ex8-nn-dynamic-batch.cfg`

| Feature | Baseline value |
| --- | ---: |
| `H` | `9.999999999999999e-06` |
| `Zn` | `1e-07` |

### EX8 Six-Feature Random-Forest Baseline

Used by `ex8-rf-6-dynamic-batch.cfg`.

| Feature | Baseline value |
| --- | ---: |
| `Zn(OH)2(aq)` | `1.813505366090861e-15` |
| `Zn(OH)3-` | `7.349990948801437e-22` |
| `Zn(OH)4--` | `2.7351872885715046e-29` |
| `Zn++` | `9.960327396019113e-08` |
| `NO3-` | `0.1000000031975435` |
| `Fe++` | `2.9324029589485202e-12` |

### EX8 Nine-Feature Random-Forest Baseline

Used by:

- `ex8-rf-9-1d.cfg`
- `ex8-rf-9-batch1.cfg`
- `ex8-rf-9-dynamic-batch.cfg`
- `ex8-rf-9-scalar.cfg`

| Feature | Baseline value |
| --- | ---: |
| `Zn(OH)2(aq)` | `1.813505366090861e-15` |
| `Zn(OH)3-` | `7.349990948801437e-22` |
| `Zn(OH)4--` | `2.7351872885715046e-29` |
| `ZnOH+` | `5.6323585057657926e-12` |
| `Zn++` | `9.960327396019113e-08` |
| `Na+` | `0.0999877981223431` |
| `NO3-` | `0.1000000031975435` |
| `Fe++` | `2.9324029589485202e-12` |
| `O2(aq)` | `0.0002464700291695` |

### LSURF Baseline

Used by:

- `lsurf-1-dynamic-batch.cfg`
- `lsurf-2-dynamic-batch.cfg`
- `lsurf-5-dynamic-batch.cfg`
- `lsurf-6-dynamic-batch.cfg`

These legacy config values remain placeholders for the LSURF inputs while PHREEQC/PhreeqcRM coupling and physical input-unit conversions are deferred. The forest expects log10 values for `uranium_total` and `Site_Density`, and log10 activities for `U_species*`. The ONNX graphs do not transform inputs.

Each LSURF ONNX graph applies `10 ** prediction` to the forest's log10 aqueous uranium prediction. The public `U_Aqueous` output is therefore linear aqueous concentration (mol/L), with the existing output tensor name and float64 type.

| Feature | Baseline value |
| --- | ---: |
| `Mineral_source` | `7` |
| `uranium_total` | `-6.677780705266080` |
| `Site_Density` | `-4.54327863489071` |
| `U_species1` | `-25.419000000000000` |
| `U_species8` | `-22.514000000000000` |
| `U_species14` | `-37.488999999999997` |
| `U_species20` | `-20.510000000000002` |

## Benchmark Cases

All benchmarks output logs to `*.out`.

### EX8 Neural-Network Benchmarks
| Config | Description | Model | Steps | 
|---|---|---|---|
| `ex8-nn-1d.cfg` | NN feature-vector inference | `ex8_nn_1d.json` | 100 |
| `ex8-nn-batch1.cfg` | NN fixed-batch inference | `ex8_nn_batch1.json` | 100 |
| `ex8-nn-dynamic-batch.cfg`| NN dynamic-batch inference | `ex8_nn_dynamic_batch.json` | 1000 |

### EX8 Random-Forest Benchmarks
| Config | Description | Model | Steps |
|---|---|---|---|
| `ex8-rf-6-dynamic-batch.cfg` | 6-feature RF inference | `ex8_rf_6_dynamic_batch.json` | 2 |
| `ex8-rf-9-1d.cfg` | 9-feature RF feature-vector | `ex8_rf_9_1d.json` | 4 |
| `ex8-rf-9-batch1.cfg` | 9-feature RF fixed-batch | `ex8_rf_9_batch1.json` | 6 |
| `ex8-rf-9-dynamic-batch.cfg` | 9-feature RF dynamic-batch | `ex8_rf_9_dynamic_batch.json` | 10 |
| `ex8-rf-9-scalar.cfg` | 9-feature RF scalar inference | `ex8_rf_9_scalar.json` | 50 |

### LSURF Benchmarks

The LSURF config and ONNX model files are optional for CTest benchmarks. If either file for a case is missing at test time, CTest reports that case as `Skipped` and continues without failing the suite. Use `ctest -V -R lsurf` to see which file is missing. Files can be added and tests rerun without reconfiguring. A model that is present but fails to load or run still fails the test. CTest 3.16 or newer is needed to display the skipped status.

| Config | Description | Model | Steps |
|---|---|---|---|
| `lsurf-1-dynamic-batch.cfg` | 1-feature RF inference | `lsurf_model_1_float_64.json` | 50 |
| `lsurf-2-dynamic-batch.cfg` | 2-feature RF inference | `lsurf_model_2_float_64.json` | 50 |
| `lsurf-5-dynamic-batch.cfg` | 5-feature RF inference | `lsurf_model_5_float_64.json` | 50 |
| `lsurf-6-dynamic-batch.cfg` | 6-feature RF inference | `lsurf_model_6_float_64.json` | 50 |

## EX8 ONNX Models & JSON Manifests

The `.json` files are Alquimia ONNX configs and the `.onnx` files are the ONNX Runtime graphs. The configs map `AlquimiaState` values into model tensors and map model outputs back into `AlquimiaState`.

### EX8 Neural-Network (`ex8_nn_*`)

Neural-network EX8 models for H/Zn [molarity] sorption.
- Input features: **H (total)**, **Zn (total)**
- Output features: **H (immobile)**, **Zn (immobile)**

| Model | Inputs | Outputs | Graph Contents | Notes |
|---|---|---|---|---|
| `ex8_nn_1d.onnx` | `chemical_input_raw: double[2]` | `sorbed_output_raw: double[2]` | Scaler math plus `MatMul`, `Add`, `Relu` | Raw H/Zn input and raw sorbed output; used by unit tests. |
| `ex8_nn_batch1.onnx` | `chemical_input_raw: double[1,2]` | `sorbed_output_raw: double[1,2]` | Scaler math plus `Gemm`, `Relu` | Fixed batch-1 raw interface. |
| `ex8_nn_dynamic_batch.onnx` | `chemical_input_raw: double[batch_size,2]` | `sorbed_output_raw: double[batch_size,2]` | Scaler math plus `Gemm`, `Relu` | Dynamic-batch raw interface. |

| Config | Model | Mapping |
|---|---|---|
| `ex8_nn_1d.json` | `ex8_nn_1d.onnx` | Reads H/Zn from `total_mobile_immobile[0]` and `[1]`; writes reactive H/Zn to `total_immobile[0]` and `[1]`. |
| `ex8_nn_batch1.json` | `ex8_nn_batch1.onnx` | Batch = 1 version of the H/Zn total-mobile-immobile to total-immobile mapping. |
| `ex8_nn_dynamic_batch.json` | `ex8_nn_dynamic_batch.onnx` | Dynamic-batch version of the H/Zn total-mobile-immobile to total-immobile mapping. |

Each config includes an `initial` condition with H = `9.999999999999999e-06` and Zn = `1e-7`.
The inference results should be H(immobile) = `-1.230666e-04`, Zn(immobile) = `-1.479950e-07`.

### EX8 Random-Forest (`ex8_rf_*`)

Random-forest EX8 models for H/Zn [molarity] sorption. The ONNX exports preserve double-precision (`float64`) inputs and outputs for Alquimia, but internally cast to single-precision (`float32`) during `Scaler` and `TreeEnsemble` opset specifically to truncate floating-point differences.

| Model | Inputs | Outputs | Graph Contents | Notes |
|---|---|---|---|---|
| `ex8_rf_6_dynamic_batch.onnx` | `double_input: double[dynamic,6]` | `double_output: double[dynamic,2]` | Scaler math, casts, `TreeEnsemble` | Six-feature forest. |
| `ex8_rf_9_batch1.onnx` | `chemical_input_raw: double[1,9]` | `sorbed_output_raw: double[1,2]` | Scaler math, casts, `TreeEnsemble` | Fixed batch-1 nine-feature forest. |
| `ex8_rf_9_dynamic_batch.onnx` | `chemical_input_raw: double[batch_size,9]` | `sorbed_output_raw: double[batch_size,2]` | Scaler math, casts, `TreeEnsemble` | Dynamic-batch nine-feature forest. |
| `ex8_rf_9_1d.onnx` | `chemical_input_raw: double[9]` | `sorbed_output_raw: double[2]` | Reshape helpers, scaler math, casts, `TreeEnsemble` | One 1-D feature vector. |
| `ex8_rf_9_scalar.onnx` | Nine `rank-0` feature tensors | `SURF-H+: double[]`, `SURF-Zn++: double[]` | Scalar packing, scaler math, casts, `TreeEnsemble` | One scalar tensor per feature. |
| `ex8_rf_9_mixed_inputs.onnx` | `chemical_dynamic: double[batch_size,3]`, `chemical_vector: double[3]`, `chemical_batch1: double[1,2]`, `chemical_scalar: double[]` | `sorbed_output_raw: double[batch_size,2]` | Mixed-rank packing, scaler math, casts, `TreeEnsemble` | Tests mixed input-rank support. |

| Config | Model | Mapping |
|---|---|---|
| `ex8_rf_6_dynamic_batch.json` | `ex8_rf_6_dynamic_batch.onnx` | Reads six aqueous species from `total_mobile[2]` through `[7]`; writes H/Zn to `total_immobile[0]` and `[1]`. |
| `ex8_rf_9_batch1.json` | `ex8_rf_9_batch1.onnx` | Reads nine aqueous species from `total_mobile[2]` through `[10]`; writes H/Zn immobile totals. |
| `ex8_rf_9_dynamic_batch.json` | `ex8_rf_9_dynamic_batch.onnx` | Same nine-feature mapping with dynamic-batch tensor shape. |
| `ex8_rf_9_1d.json` | `ex8_rf_9_1d.onnx` | Same nine-feature mapping with a 1-D feature-vector tensor. |
| `ex8_rf_9_scalar.json` | `ex8_rf_9_scalar.onnx` | Maps each feature to its own rank-0 tensor and reads scalar H/Zn outputs. |
| `ex8_rf_9_mixed_inputs.json` | `ex8_rf_9_mixed_inputs.onnx` | Splits nine features across dynamic, vector, batch-1, and scalar input tensors. |

The nine-feature configs use: `Zn(OH)2(aq)`, `Zn(OH)3-`, `Zn(OH)4--`, `ZnOH+`, `Zn++`, `Na+`, `NO3-`, `Fe++`, and `O2(aq)`.

All random-forest configs include an `initial` condition for the mapped aqueous species and write model outputs to:

| Output | Alquimia Destination | Unit |
|---|---|---|
| H sorbed amount | `total_immobile[0]` | moles/m^3 bulk |
| Zn sorbed amount | `total_immobile[1]` | moles/m^3 bulk |
