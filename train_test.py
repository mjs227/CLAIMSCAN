
import gc
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
import torch.optim as op
from models import ClaimscanModel
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer


FP = '/path/to/files/'
EPOCHS = 15
OPTIMIZER = op.Adam
OPTIMIZER_KWARGS = {'lr': 1e-4, 'weight_decay': 1e-5}
OPTIMIZER_LM = None  # defaults to OPTIMIZER
OPTIMIZER_LM_KWARGS = {'lr': 1e-5, 'weight_decay': 1e-6}
HEAD_P = 0.25
LM_DIM = 768
L_WINDOW = 7
R_WINDOW = 7
PT_KWARGS = {
    'd_model': 5,
    'n_layers': 3,
    'p': 0.1
}
RANDOM_SEED = 25  # 6
AGGR_TYPE = 'avg'
BOS, EOS = torch.tensor([[0]]), torch.tensor([[2]])
DEVICE = 0 if torch.cuda.is_available() else 'cpu'


def get_spans(model_pred_list):
    in_span, span_starts, span_ends = False, [], []

    for i in range(len(model_pred_list)):
        if model_pred_list[i] == 1:
            if not in_span:
                span_starts.append(i)
                in_span = True
        elif in_span:
            span_ends.append(i - 1)
            in_span = False

    if in_span:
        span_ends.append(len(model_pred_list) - 1)

    assert len(span_starts) == len(span_ends)

    return str(span_starts), str(span_ends)


random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
lm = AutoModel.from_pretrained('roberta-base')

model = ClaimscanModel(
    lm,
    optimizer=OPTIMIZER,
    optimizer_lm=OPTIMIZER_LM,
    optimizer_kwargs=OPTIMIZER_KWARGS
    optimizer_lm_kwargs=OPTIMIZER_LM_KWARGS
    lm_dim=LM_DIM,
    head_p=HEAD_P,
    l_window=L_WINDOW,
    r_window=R_WINDOW,
    pt_kwargs=PT_KWARGS
)

with open(FP + 'Task_B/train.json', 'r') as f:
    train_json = json.load(f)

with open(FP + 'Task_B/dev.json', 'r') as f_b:
    dev = json.load(f)

x_dev_b, y_dev, train_data = [], [], []

for item in tqdm(dev):
    tokens_list, targets_list, trgt_ten_list, spans = [BOS.clone()], [], [], set()
    item_tokens = item['tokens']
    token_spans, span_idx = [], 0

    for a, b in map(tuple, item['spans']):
        spans = spans.union(set(range(a, b + 1)))

    for i in range(len(item['tokens'])):
        tok = tokenizer([item['tokens'][i]], return_tensors='pt')
        tokens_list.append(tok['input_ids'][:, 1:-1])
        trgt_lbl = 1 if i in spans else 0
        targets_list.append(trgt_lbl)
        trgt_ten_list += [trgt_lbl] * tokens_list[-1].size(1)
        span_idx += tokens_list[-1].size(1)
        token_spans.append(span_idx)

    tokens_list.append(EOS.clone())
    x_dev.append((torch.cat(tokens_list, dim=1), token_spans))
    y_dev.append(targets_list)

for item in tqdm(train_json):
    tokens_list, targets_list, spans = [BOS.clone()], [], set()

    for a, b in map(tuple, item['spans']):
        spans = spans.union(set(range(a, b + 1)))

    for i in range(len(item['tokens'])):
        tok = tokenizer([item['tokens'][i]], return_tensors='pt')
        tokens_list.append(tok['input_ids'][:, 1:-1])
        targets_list += [1 if i in spans else 0] * tokens_list[-1].size(1)

    tokens_list.append(EOS.clone())
    train_data.append((torch.cat(tokens_list, dim=1), torch.tensor([targets_list], dtype=torch.float32), has_span))

best_state_dict = None
model.to(DEVICE)
model.zero_grad()

loss_fn = nn.BCEWithLogitsLoss()
best_f1, patience_cnt = 0, 0
random.shuffle(train_data)

for ep in range(EPOCHS):
    if patience_cnt < 5:
        break
    
    model.train()
    ep_loss = 0

    print(f'Epoch {ep}:')
    print()

    for x_train, y_train in tqdm(train_data):
        span_pred = model(x_train.to(DEVICE))
        loss = loss_fn(span_pred, y_train.to(DEVICE))
        loss.backward()
        model.step()
        model.zero_grad()

        ep_loss += loss.item()

    print()
    print(f'    Train loss: {round(ep_loss / len(train_data), 5)}')
    print()

    model.eval()
    preds_avg, preds_max, preds_min = [], [], []

    with torch.no_grad():
        for i in tqdm(range(len(x_dev))):
            model_input, token_spans = x_dev[i]
            model_pred = model(model_input.to(DEVICE))
            span_preds_avg, span_preds_max, span_preds_min = [], [], []
            span_idx = 0

            for s in token_spans:
                span_segment = model_pred[span_idx:s]
                span_preds_avg.append(torch.round(torch.sum(span_segment) / span_segment.size(0)).item())
                span_preds_max.append(torch.round(torch.max(span_segment)).item())
                span_preds_min.append(torch.round(torch.min(span_segment)).item())
                span_idx = s

            preds_avg.append(span_preds_avg)
            preds_max.append(span_preds_max)
            preds_min.append(span_preds_min)

    f1_avg = sum(f1_score(l, p, average='macro', zero_division=0) for l, p in zip(y_dev_b, preds_b_avg))
    f1_max = sum(f1_score(l, p, average='macro', zero_division=0) for l, p in zip(y_dev_b, preds_b_max))
    f1_min = sum(f1_score(l, p, average='macro', zero_division=0) for l, p in zip(y_dev_b, preds_b_min))
    f1_avg /= len(y_dev_b)
    f1_max /= len(y_dev_b)
    f1_min /= len(y_dev_b)
    f1 = max(f1_avg, f1_max, f1_min)

    if best_f1 > f1:
        patience_cnt += 1
    else:
        best_state_dict, patience_cnt = model.state_dict(), 0

    best_f1 = max(best_f1, f1)

    print()
    print(f'    Eval. F1 Score (Avg.): {f1_avg}')
    print(f'    Eval. F1 Score (Max.): {f1_max}')
    print(f'    Eval. F1 Score (Min.): {f1_min}')
    print(f'    Patience: {patience_cnt}')
    print()
    print()

    gc.collect()

    if DEVICE == 0:
        torch.cuda.empty_cache()

if best_state_dict is not None:
    torch.save(best_state_dict, FP + 'best_state_dict.pt')

with open(FP + '/Task_B/test.json', 'r') as f:
    test_json = json.load(f)

test_data = []

for item_tokens in tqdm(test_json):
    tokens_list, token_spans, span_idx = [BOS.clone()], [], 0

    for i in range(len(item_tokens)):
        tok = tokenizer([item_tokens[i]], truncation=True, return_tensors='pt')
        tokens_list.append(tok['input_ids'][:, 1:-1])
        span_idx += tokens_list[-1].size(1)
        token_spans.append(span_idx)

    tokens_list.append(EOS.clone())
    test_data.append((torch.cat(tokens_list, dim=1), token_spans))

model.eval()
model.load_state_dict(best_state_dict)
csv_file = pd.read_csv(FP + 'Task_B/test.csv')
csv_file['span_start_index'] = [''] * len(csv_file)
csv_file['span_end_index'] = [''] * len(csv_file)

if AGGR_TYPE == 'avg':
    aggr_fn = lambda x: torch.sum(x) / x.size(0)
elif AGGR_TYPE == 'max':
    aggr_fn = lambda x: torch.max(x)
elif AGGR_TYPE == 'min':
    aggr_fn = lambda x: torch.min(x)
else:
    raise NotImplementedError

with torch.no_grad():
    for i in tqdm(range(len(test_data))):
        model_input, token_spans = test_data[i]
        model_pred = torch.sigmoid(model(model_input.to(DEVICE)).flatten())
        span_idx, span_preds = 0, []

        for s in token_spans:
            span_preds.append(int(torch.round(aggr_fn(model_pred[span_idx:s])).item()))
            span_idx = s

        start_indices, end_indices = get_spans(span_preds)
        csv_file.loc[i, 'span_start_index'] = start_indices
        csv_file.loc[i, 'span_end_index'] = end_indices

csv_file.to_csv(FP + 'test_res.csv', index=False)
