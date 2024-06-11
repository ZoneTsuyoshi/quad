import argparse, datetime, os, shutil, json, glob
from distutils.util import strtobool
from typing import Optional, Dict, Any, List
import logtools


def get_parser_for_generating_data():
    parser = argparse.ArgumentParser(description="Generate data for the QuADNet model")

    # overall parameters
    parser.add_argument("-dd", "--data_dir", type=str, default="../data", help="Directory to save the generated data")
    parser.add_argument("-dt", "--data_type", type=str, default="qp5", help="Type of data to generate")
    parser.add_argument("-s", "--seed", type=int, default=128, help="Random seed")
    parser.add_argument("-d", "--dim", type=int, default=4, help="Dimension of the data")
    parser.add_argument("-std", "--std", type=float, default=0.4, help="Standard deviation of the noise")
    parser.add_argument("-ap", "--anomaly_probability", type=float, default=0.2, help="Probability of anomaly")

    # synthetic quasi-periodic data parameters
    parser.add_argument("-np", "--n_periods", type=int, default=500, help="Number of periods of the synthetic data")
    parser.add_argument("-bp", "--base_period", type=int, default=90, help="Base period of the synthetic data")
    parser.add_argument("-pf", "--period_fluctuation", type=float, default=0.5, help="Period fluctuation of the synthetic data")
    parser.add_argument("-cpd", "--cutoff_period_difference", type=int, default=2, help="Cutoff period difference of the synthetic data")
    parser.add_argument("-nw", "--n_waves", type=int, default=10, help="Number of waves in the synthetic data")
    parser.add_argument("-apb", "--anomaly_period_bias", type=int, default=5, help="Bias of the anomaly period")
    parser.add_argument("-acp", "--anomaly_continuous_probability", type=float, default=0.3, help="Probability of continuous anomaly")

    return parser


def get_parser_for_training():
    parser = argparse.ArgumentParser(description="Train the QuADNet model")

    # overall parameters
    parser.add_argument("-dd", "--data_dir", type=str, default="../data/qp5", help="Directory to load the data")
    parser.add_argument("-rd", "--result_dir", type=str, default=None, help="Directory to save the results")
    parser.add_argument("-s", "--seed", type=int, default=128, help="Random seed")
    parser.add_argument("-g", "--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("-l", "--log", type=str, default="CometLogger", choices=["EmptyLogger", "CometLogger", "NeptuneLogger"], help="Logger type")
    parser.add_argument("-nolog", "--no_logger", action="store_true", help="Do not use logger")
    parser.add_argument("-ek", "--experiment_key", type=str, default=None, help="Experiment key")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-t", "--test", action="store_true", help="Test mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    # data parameters
    parser.add_argument("-ws", "--window_size", type=int, default=120, help="Window size")
    parser.add_argument("-hs", "--horizon", type=int, default=10, help="Horizon")
    parser.add_argument("-dst", "--data_stride", type=int, default=1, help="Stride")
    parser.add_argument("-sc", "--scaling", type=str, default="S", choices=["S", "M", "R", "N"], help="Scaling method")
    parser.add_argument("-vr", "--valid_ratio", type=float, default=0.2, help="Validation ratio")

    # training parameters
    parser.add_argument("-c", "--criterion", type=str, default="MSELoss", help="Criterion")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-ne", "--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-opt", "--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-mgn", "--max_grad_norm", type=float, default=1.0, help="Max grad norm")
    parser.add_argument("-neval", "--n_evaluate_every", type=int, default=1, help="Evaluate every")
    parser.add_argument("-ctw", "--cutoff_time_window", type=int, default=12, help="Cutoff time window")
    parser.add_argument("-sw", "--smoothed_window", type=int, default=90, help="Smoothed window")

    # model parameters
    parser.add_argument("-hn", "--rnn_hidden", type=int, default=16, help="RNN hidden size")
    parser.add_argument("-cc", "--cnn_channels", type=int, default=16, help="CNN channels")
    parser.add_argument("-ah", "--attn_hidden", type=int, default=16, help="Attention hidden size")
    parser.add_argument("-nl", "--n_rnn_layers", type=int, default=1, help="Number of RNN layers")
    parser.add_argument("-nc", "--n_cnn_layers", type=int, default=1, help="Number of CNN layers")
    parser.add_argument("-no", "--n_output_layers", type=int, default=1, help="Number of output layers")
    parser.add_argument("-af", "--activation_function", type=str, default="SiLU", help="Activation function")
    parser.add_argument("-ks", "--kernel_size", type=int, default=3, help="Kernel size")
    parser.add_argument("-st", "--stride", type=int, default=1, help="Stride")
    parser.add_argument("-pd", "--padding", type=int, default=0, help="Padding")
    parser.add_argument("-dl", "--dilation", type=int, default=1, help="Dilation")
    parser.add_argument("-bn", "--use_batchnorm", type=strtobool, default="false", help="Use batch normalization")
    parser.add_argument("-cdr", "--cnn_dropout", type=float, default=0., help="CNN dropout")
    parser.add_argument("-rt", "--rnn_type", type=str, default="GRU", help="RNN type")
    parser.add_argument("-rdr", "--rnn_dropout", type=float, default=0., help="RNN dropout")
    parser.add_argument("-bd", "--bidirectional", type=strtobool, default="false", help="Bidirectional RNN")
    parser.add_argument("-at", "--attn_type", type=str, default="simple", choices=["raw", "simple", "multihead"], help="Attention type")
    parser.add_argument("-nh", "--n_heads", type=int, default=1, help="Number of heads")
    parser.add_argument("-adr", "--attn_dropout", type=float, default=0., help="Attention dropout")
    parser.add_argument("-uls", "--use_last_stat", type=strtobool, default="false", help="Use last stat")

    return parser


def set_result_direcotry(debug: bool = False, result_dir: Optional[str] = None, grid_search_on: bool = False):
    if result_dir is None:
        dt_now = datetime.datetime.now()
        upper_dir = "../results"

        if debug:
            name = "d"
            if os.path.exists(os.path.join(upper_dir, "d")):
                shutil.rmtree(os.path.join(upper_dir, "d"))
        else:
            name = "{}_".format(dt_now.strftime("%y%m%d"))
            if grid_search_on:
                name += "gs"
            i = 1
            while os.path.exists(os.path.join(upper_dir, name + str(i))):
                i += 1
            name += str(i)

        result_dir = os.path.join(upper_dir, name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    return result_dir


def initialize(args: argparse.Namespace) -> argparse.Namespace:
    """Convert args whether test or not."""
    experiment_name = args.data_dir.split("/")[-1]
    logger = getattr(logtools, args.log)(experiment_name, args.experiment_key)
    if args.test:
        config = vars(args)
        with open(os.path.join(args.result_dir, "config.json"), "r") as f:
            config.update(json.load(f))
        args = argparse.Namespace(**config)
        args.test = True
    else:
        args = convert_None_arguments_to_Nones(args)
        args.result_dir = set_result_direcotry(args.debug, args.result_dir)
        if args.debug:
            args.n_epochs = 1
        if args.no_logger:
            args.logger = "EmptyLogger"
        save_config(args)
        logger.log_parameters(vars(args))
    return args, logger


def convert_None_arguments_to_Nones(args: argparse.Namespace) -> argparse.Namespace:
    """Convert None arguments to default values."""
    for key, value in vars(args).items():
        if value == "None":
            setattr(args, key, None)
    return args


def save_config(args: argparse.Namespace):
    """Save config file."""
    with open(os.path.join(args.result_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)