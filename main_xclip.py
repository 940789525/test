from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_xclip import XCLIP
from modules.optimization import BertAdam
from util import get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

try:
    from modules.lora import inject_lora
    _LORA_AVAILABLE = True
except ImportError:
    print("LoRA modules not found. Please ensure 'lora.py' and 'modules/__init__.py' exist.")
    _LORA_AVAILABLE = False
    def inject_lora(model, r):
        pass

torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='X-CLIP on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    # --- 保留所有原有的参数解析 ---
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2], help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"], help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP", choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"], help="choice a similarity header.")
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument('--use_lora', action='store_true', help="Whether to use LoRA for fine-tuning.")
    parser.add_argument('--lora_rank', type=int, default=8, help="Rank for LoRA matrices.")
    parser.add_argument('--lora_alpha', type=int, default=16, help="Alpha for LoRA scaling.")
    parser.add_argument('--lora_dropout', type=float, default=0.0, help="Dropout for LoRA layers.")
    parser.add_argument('--lora_target_modules', nargs='+', type=str, default=None, help="Not used by new lora.py.")
    parser.add_argument('--lora_bias', type=str, default='none', help="Bias type for LoRA layers (e.g., 'none', 'all'). Not used by new lora.py but kept for compatibility.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2], help="Frame order for evaluation. 0: head, 1: tail, 2: uniform. Default is 0 for deterministic eval.")
    parser.add_argument('--train_frame_order', type=int, default=2, choices=[0, 1, 2], help="Frame order for training. 0: head, 1: tail, 2: uniform. Default is 2 to match slice_framepos.")

    args = parser.parse_args()

    if args.use_lora and not _LORA_AVAILABLE:
        raise ImportError("LoRA modules could not be imported correctly.")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1")
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    args.rank = torch.distributed.get_rank()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info(f"  <<< {key}: {args.__dict__[key]}")

    return args

def init_device(args, local_rank):
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = torch.cuda.device_count()
    logger.info(f"device: {device} n_gpu: {n_gpu}")
    args.n_gpu = n_gpu
    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter.")
    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = XCLIP.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    model.to(device)
    return model

def prep_optimizer(args, model, num_train_optimization_steps):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
         logger.warning("No trainable parameters found for the optimizer!")
         return None, None, model

    optimizer = BertAdam(trainable_params, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=0.2,
                         max_grad_norm=1.0)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=False)
    return optimizer, None, model

def save_model(epoch, args, model, optimizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    if args.use_lora:
        lora_state_dict = {k: v for k, v in model_to_save.state_dict().items() if 'lora' in k}
        output_model_file = os.path.join(args.output_dir, f"pytorch_lora_weights.ep{epoch}.bin")
        torch.save(lora_state_dict, output_model_file)
        logger.info(f"LoRA weights saved to {output_model_file}")
    else:
        output_model_file = os.path.join(args.output_dir, f"pytorch_model.ep{epoch}.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info(f"Full model saved to {output_model_file}")
    return output_model_file

def train_epoch(epoch, args, model, train_dataloader, device, optimizer, global_step):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, video, video_mask = batch
        loss = model(input_ids, segment_ids, input_mask, video, video_mask)
        if args.n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % log_step == 0 and args.local_rank == 0:
                elapsed_time = time.time() - start_time
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Step {step+1}/{len(train_dataloader)}, "
                            f"Lr: {optimizer.get_lr()[0]:.8f}, Loss: {loss.item():.4f}, "
                            f"Time/step: {elapsed_time / (log_step * args.gradient_accumulation_steps):.2f}s")
                start_time = time.time()
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_seq_features_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        seq_features = batch_seq_features_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, seq_features, visual_output, input_mask, video_mask,
                                                                     loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def eval_epoch(args, model, test_dataloader, device):
    model_eval = model.module if hasattr(model, 'module') else model
    model_eval.eval()
    with torch.no_grad():
        batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list, batch_seq_features_list = [], [], [], [], []
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch
            (sequence_output, seq_features), visual_output = model_eval.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)
            batch_sequence_output_list.append(sequence_output)
            batch_seq_features_list.append(seq_features)
            batch_list_t.append((input_mask, segment_ids,))
            batch_visual_output_list.append(visual_output)
            batch_list_v.append((video_mask,))
            if args.local_rank == 0: print(f"{bid}/{len(test_dataloader)}\r", end="")
        
        sim_matrix = _run_on_single_gpu(model_eval, batch_list_t, batch_list_v, batch_sequence_output_list, batch_seq_features_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    
    if args.local_rank == 0:
        logger.info(f"sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info("Text-to-Video:")
        logger.info(f"\t>>>  R@1: {tv_metrics['R1']:.1f} - R@5: {tv_metrics['R5']:.1f} - R@10: {tv_metrics['R10']:.1f} - Median R: {tv_metrics['MR']:.1f} - Mean R: {tv_metrics['MeanR']:.1f}")
        logger.info("Video-to-Text:")
        logger.info(f"\t>>>  V2T$R@1: {vt_metrics['R1']:.1f} - V2T$R@5: {vt_metrics['R5']:.1f} - V2T$R@10: {vt_metrics['R10']:.1f} - V2T$Median R: {vt_metrics['MR']:.1f} - V2T$Mean R: {vt_metrics['MeanR']:.1f}")
    
    return tv_metrics['R1'] if args.local_rank == 0 else 0

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()
    model = init_model(args, device, n_gpu, args.local_rank)

    if args.use_lora:
        logger.info("Applying LoRA fine-tuning with explicit initial layer freezing.")
        
        # <<<< 明确指定需要冻结的 CLIP 初始层 >>>>
        clip_initial_layers_to_freeze = [
            "clip.positional_embedding",
            "clip.visual.class_embedding",
            "clip.visual.positional_embedding",
            "clip.visual.conv1.weight",
            "clip.visual.ln_pre.weight",
            "clip.visual.ln_pre.bias",
            "clip.token_embedding.weight"
        ]
        
        # 注入 LoRA 层
        inject_lora(model.clip, r=args.lora_rank)

        # 遍历所有参数，应用冻结/解冻规则
        for name, param in model.named_parameters():
            # 规则 1：默认冻结所有参数
            param.requires_grad = False

            # 规则 2：解冻 LoRA 参数
            if 'lora' in name:
                param.requires_grad = True

            # 规则 3：解冻分类头
            if 'loose_type' in name:
                param.requires_grad = True
        
        # 规则 4: 再次遍历，确保初始层被强制冻结 (即使名字中包含 'loose_type' 等, 虽然不太可能)
        for name, param in model.named_parameters():
             # DDP 会在参数名前加上 'module.'，我们需要去掉它来匹配列表
            clean_name = name.replace('module.', '')
            if clean_name in clip_initial_layers_to_freeze:
                param.requires_grad = False
        model.to(device)
        if args.local_rank == 0:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA injected. Total parameters: {total_params:,}. Trainable: {trainable_params:,}")
            logger.info("--- Frozen Initial Layers ---")
            for layer_name in clip_initial_layers_to_freeze:
                logger.info(f"  - {layer_name}")
            logger.info("-----------------------------")


    # --- 数据加载和训练/评估 ---
    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1) / args.gradient_accumulation_steps) * args.epochs
        
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info(f"  Num train examples = {train_length}")
            logger.info(f"  Batch size = {args.batch_size}")
            logger.info(f"  Total optimization steps = {num_train_optimization_steps}")
        
        global_step = 0
        best_r1 = 0
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, optimizer, global_step)
            
            if args.local_rank == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs} Finished, Train Loss: {tr_loss:.4f}")
                saved_model_path = save_model(epoch, args, model, optimizer)
                
                if test_dataloader is not None:
                    r1 = eval_epoch(args, model, test_dataloader, device)
                    if r1 > best_r1:
                        best_r1 = r1
                        logger.info(f"🎉 New best R@1: {best_r1:.2f} at epoch {epoch+1}. Model saved to {saved_model_path}")
    
    elif args.do_eval:
        if args.local_rank == 0:
            if not args.init_model:
                logger.error("For do_eval, --init_model must be specified.")
                return
            
            if args.use_lora:
                logger.info(f"Loading LoRA weights for evaluation from: {args.init_model}")
                lora_weights = torch.load(args.init_model, map_location='cpu')
                model.load_state_dict(lora_weights, strict=False)

            logger.info(f"Evaluating model: {args.init_model}")
            eval_epoch(args, model, test_dataloader, device)

if __name__ == "__main__":
    main()