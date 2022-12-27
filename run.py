from deepbills.args_parser import create_args_parser
from deepbills.model import DeepBills
from deepbills.bill_dataset import load_bill_dataset

parser = create_args_parser()
args   = parser.parse_args()

dataset    = load_bill_dataset(args.problem_name, args.position)

Bill_model = DeepBills(dataset   = dataset, 
                    model_name   = args.model_name,
                    fold_name    = args.fold_n,
                    problem_name = args.problem_name, 
                    batch_size   = args.batch_size, 
                    epoch        = args.n_epoch, 
                    use_fp16     = args.use_fp16)

Bill_model.trainer.train()