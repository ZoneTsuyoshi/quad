import os, argparse, inspect
import numpy as np
import torch
import data, parse, models, train_tools


def main(args: argparse.Namespace, logger):
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.gpu_id)

    # load data
    train_inputs, train_outputs, valid_inputs, valid_outputs, test_inputs, test_outputs, test_periods, test_labels = data.load_data(args.data_dir, args.window_size, args.horizon, args.data_stride, args.scaling, args.valid_ratio)
    n_channels = train_inputs.shape[-1]

    # prepare model
    model_args = inspect.getfullargspec(models.QuADNet).args
    model_args = {k: v for k, v in vars(args).items() if k in model_args}
    model = models.QuADNet(n_channels, **model_args).to(device)
    print(model)

    # prepare trainer
    trainer = train_tools.Trainer(model, device, logger, args.result_dir, train_inputs, train_outputs, valid_inputs, valid_outputs, args.criterion, args.batch_size, args.n_epochs, args.optimizer, args.learning_rate, args.max_grad_norm, args.n_evaluate_every, args.cutoff_time_window, args.smoothed_window, args.use_reconstruction)

    # train
    if args.test:
        trainer.load()
        print(trainer.train_period_class, trainer.train_class_probs)
    else:
        trainer.train(args.verbose)

    # test
    trainer.test(test_inputs, test_outputs, test_labels, test_periods, args.verbose)


if __name__ == "__main__":
    parser = parse.get_parser_for_training()
    args = parser.parse_args()
    args, logger = parse.initialize(args)
    main(args, logger)