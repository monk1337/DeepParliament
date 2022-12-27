import argparse


def create_args_parser():
    parser = argparse.ArgumentParser(description="DeepBills for Parliament Bill Prediction")
    parser.add_argument("--problem_name", type=str, default="multiclass", required=False,
                        help="The problem name [multiclass or binary]")

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--fold_n', type=int, default=5)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Choose the model_name from hugginface repo which can be bert-base-uncase etc")
    
    parser.add_argument("--use_fp16", action='store_true', default=False)
    
    return 