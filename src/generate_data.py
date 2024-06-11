import os, argparse, json
from parse import get_parser_for_generating_data
from utils_gen import *

def generate_data(args: argparse.Namespace):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    with open(os.path.join(args.data_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.data_type == "qp5":
        data_with_anomalies, periods_with_anomalies, labels = generate_quasi_periodic_five(args.dim, args.n_periods, args.base_period, args.period_fluctuation, args.cutoff_period_difference, args.n_waves, args.anomaly_period_bias, args.anomaly_continuous_probability, args.std, args.anomaly_probability, args.seed)
        data_without_anomalies, periods_without_anomalies, _ = generate_quasi_periodic_five(args.dim, args.n_periods, args.base_period, args.period_fluctuation, args.cutoff_period_difference, args.n_waves, 0, 0, args.std, 0, args.seed)

    np.save(os.path.join(args.data_dir, "test_data.npy"), data_with_anomalies)
    np.save(os.path.join(args.data_dir, "test_periods.npy"), periods_with_anomalies)
    np.save(os.path.join(args.data_dir, "test_labels.npy"), labels)
    np.save(os.path.join(args.data_dir, "train_data.npy"), data_without_anomalies)
    np.save(os.path.join(args.data_dir, "train_periods.npy"), periods_without_anomalies)


if __name__ == "__main__":
    parser = get_parser_for_generating_data()
    args = parser.parse_args()
    generate_data(args)