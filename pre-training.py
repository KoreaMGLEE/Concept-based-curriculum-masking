import os
import copy
import random
import argparse
import sys
import os
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from src.bert_layers import Bert_For_Att_output_MLM
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from configuration.config import Bert_Medium_Config
from src.dataloader import padded_sequence, create_dataset_base_dynamic


def get_disc_batch(logits, sub_batch, sub_label_position):
    batch_size = logits.shape[0]
    disc_batch = copy.deepcopy(sub_batch)

    for example_idx in range(batch_size):  # 먼저 batch index로 돌아

        masked_idx = sub_label_position[example_idx]
        pred = logits[example_idx, masked_idx, :]

        uniform_noise = torch.rand(pred.size(), device=device)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-9) + 1e-9).to(torch.float32)

        replaced_tokens = torch.argmax(torch.softmax(pred + gumbel_noise, dim=-1), dim=-1)

        disc_batch[example_idx, masked_idx] = replaced_tokens

    return disc_batch, sub_label_position


def train(model, summary, args):
    scaler = torch.cuda.amp.GradScaler()

    step = 1
    iters = 1


    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(-1)

    curriculum_num = 0
    for epoch in range(args.epochs):
        Loss = 0
        Lm_Loss = 0
        Loss_len = 0

        Cm_number = 0
        Total_number = 0
        passed_example = 0

        print("now %s epoch..." % str(epoch + 1))
        folders = os.listdir(args.data_path)
        random.shuffle(folders)
        for file in folders:
            train_dataset = create_dataset_base_dynamic(args.data_path, file, curriculum_num, tokenizer)
            train_dataloader = DataLoader(train_dataset, batch_size=args.step_batch_size, shuffle=True,
                                          collate_fn=padded_sequence, drop_last=True, num_workers=10)

            for batch in tqdm(train_dataloader, ncols=100):
                lm_embed, lm_label_embed, label_mask_, label_position, concept_mask_count, total_mask_count = batch

                length = [len(l) for l in label_position]
                if min(length) < 3:
                    passed_example += 1
                    continue

                for i in range(int(args.step_batch_size / args.batch_size)):
                    sub_lm_embed = lm_embed[i * args.batch_size:(i + 1) * args.batch_size]
                    sub_lm_label = lm_label_embed[i * args.batch_size:(i + 1) * args.batch_size]
                    sub_label_mask = label_mask_[i * args.batch_size:(i + 1) * args.batch_size]
                    sub_batch = torch.LongTensor(sub_lm_embed).cuda()
                    sub_lm_label = torch.LongTensor(sub_lm_label).cuda()
                    sub_label_mask = torch.BoolTensor(sub_label_mask).cuda()

                    attention_mask_ = (sub_batch == tokenizer.pad_token_id)
                    zero_pad = torch.zeros(attention_mask_.size()).cuda()
                    _attention_mask_ = zero_pad.masked_fill(attention_mask_, 1)

                    with torch.cuda.amp.autocast():
                        outputs = model(sub_batch, attention_mask=_attention_mask_)
                        logits = outputs[0]

                        lm_label = sub_lm_label.masked_fill(sub_label_mask, -100)
                        lm_loss = criterion(logits.view(-1, 30522), lm_label.view(-1))

                        loss = lm_loss / (args.step_batch_size / args.batch_size)
                        loss = loss.mean()

                        Loss += loss.item()
                        Lm_Loss += lm_loss.item() / (args.step_batch_size / args.batch_size)

                        scaler.scale(loss).backward()

                lr_scheduler.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


                iters += 1

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                Loss_len += 1

                Cm_number += sum(concept_mask_count) / args.step_batch_size
                Total_number += sum(total_mask_count) / args.step_batch_size

                if iters % 100 == 0:
                    summary.add_scalar('loss/loss_a', float((Loss + 1e-6) / (Loss_len + 1e-5)), step)
                    summary.add_scalar("loss/lm_loss", float((Lm_Loss + 1e-6) / (Loss_len + 1e-5)), step)
                    summary.add_scalar("hyp_para/cm_mask", float(Cm_number / (Loss_len + 1e-5)), step)
                    summary.add_scalar("hyp_para/total_mask", float(Total_number / (Loss_len + 1e-5)), step)
                    Loss = 0
                    Loss_len = 0
                    Lm_Loss = 0

                    Cm_number = 0
                    Total_number = 0

                if iters % 100000 == 0:
                    PATH = '../'
                    torch.save(model.state_dict(), PATH)
                    curriculum_num += 1
                    if curriculum_num == 5:
                        curriculum_num = 0
                        break
                    else:
                        break
                if iters == 1000000:
                    quit()

    summary.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", default='0', help="choose gpu number: 0, 1, 2, 3", type=int)
    parser.add_argument("--pretrained", default='bert-base_uncased',
                        help="choose model pretrained weight from: bert-base-uncased, bert-large-uncased, roberta-base, roberta-large",
                        type=str)
    parser.add_argument("--lr", default=5e-4, help="insert learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.01, help="insert weight decay", type=float)
    parser.add_argument("--epochs", default=1000, help="insert epochs", type=int)
    parser.add_argument("--batch_size", default=128, help="insert batch size", type=int)
    parser.add_argument("--step_batch_size", default=128, help="insert step batch size", type=int)
    parser.add_argument("--random_seed", default=16, help="insert step batch size", type=int)
    parser.add_argument("--data_path", default='../',
                        type=str)
    parser.add_argument("--warmup_steps", default=100000,
                        type=int)

    args = parser.parse_args()
    args.random_seed = random.randint(1, 5000)
    summary = SummaryWriter(comment='runs/BERT_%s_%s' % (str(args.pretrained), str(args.random_seed)))
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    configuration = Bert_Medium_Config
    model = Bert_For_Att_output_MLM(configuration, True, 128)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-6, lr=args.lr)
    optimizer.zero_grad()
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=1000000)
    model.to(device)


    train(model, summary, args)