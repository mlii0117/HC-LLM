import os
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.R2GenGPT import R2GenGPT
from lightning.pytorch import seed_everything
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer,AutoModelForCausalLM
import lightning.pytorch as pl
import os
# os.environ['MASTER_PORT'] = '43309'

# import os
# import torch.distributed as dist
#
# # #Set the RANK environment variable
# os.environ['RANK'] = '4'  #Replace with the actual rank of the process
# #
# # #Set the WORLD_SIZE environment variable
# os.environ['WORLD_SIZE'] = '4'  #Replace with the actual world size
#
# #Set the MASTER_ADDR environment variable
# os.environ['MASTER_ADDR'] = '127.0.0.1'  #Replace with the actual master address
#
# #Set the MASTER_PORT environment variable
# os.environ['MASTER_PORT'] = '6037'  #Replace with the actual master port
#
# #Initialize the distributed process group
# dist.init_process_group(backend='nccl', init_method='env://')
# import datetime
#
# timeout = datetime.timedelta(minutes=30)  #30 minutes timeout
# dist.init_process_group(backend='nccl', init_method='env://', timeout=timeout)
from lightning.pytorch.strategies import DDPStrategy
# from pytorch_lightning.strategies import DDPStrategy
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  #or "true"

def train(args):

    dm = DataModule(args)
    callbacks = add_callbacks(args)
    # 检查并设置 find_unused_parameters
    # if args.strategy == 'ddp':
    #     strategy = DDPStrategy(find_unused_parameters=True)
    # else:
    #     strategy = args.strategy
    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy='ddp_find_unused_parameters_true',
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    if args.ckpt_file is not None:
        model = R2GenGPT.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = R2GenGPT(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model_id = "/HUyongli/fly/code/data/BiMediX-Bi/"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # embed_tokens = model.get_input_embeddings()
    # print(type(embed_tokens))
    # text = "Hello BiMediX! I've been experiencing increased tiredness in the past week."
    # inputs = tokenizer(text, return_tensors="pt")
    # outputs = model.generate(**inputs, max_new_tokens=500)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    train(args)


if __name__ == '__main__':
    main()