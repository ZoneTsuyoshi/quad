# Repository for Quasi-periodic Anomaly Detection (QuAD)
The QuAD system proposed in [the paper](https://ieeexplore.ieee.org/document/9969061) detects anomalies in time series based on representations learned from time series prediction.
The core of the system is a time series prediction network, QuADNet.
This repository contributes to the ease of use of QuADNet.

## How to Use
We recommend that you check `src/run.py` first.
The file is trained and tested by QuADNet based on the variables received by the parser.
The list of variables received by parser can be viewed by running `python run.py -h` or by checking the `src/parse.py` file.
Data should be prepared in data_dir under the names `train_data.npy` and `test_data.npy`.
The `train_data.npy` file is loaded during training, and the `test_data.npy` file is loaded during testing.

## Citation
```bibtex
@INPROCEEDINGS{iecon-quad-2022,
  author={Ishizone, Tsuyoshi and Higuchi, Tomoyuki and Okusa, Kosuke and Nakamura, Kazuyuki},
  booktitle={IECON 2022 – 48th Annual Conference of the IEEE Industrial Electronics Society}, 
  title={An Online System of Detecting Anomalies and Estimating Cycle Times for Production Lines}, 
  year={2022},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Meters;Energy consumption;Power demand;Neural networks;Estimation;Benchmark testing;anomaly detection;key production performance indicator;quasi-periodicity;attention mechanism;smart meter},
  doi={10.1109/IECON49645.2022.9969061}}
```