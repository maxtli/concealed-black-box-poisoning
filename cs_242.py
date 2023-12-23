# %%
import torch
from tqdm import tqdm
import torch.optim
import seaborn as sns
from training_utils import load_model_data, save_hook_last_token, ablation_hook_last_token

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "gpt2-small"
batch_size = 50
ctx_length = 50
# device, model, tokenizer, owt_loader = load_model_data(model_name, batch_size, ctx_length, ds_name="maxtli/OpenWebText-2M", repeats=False)
device, model, tokenizer, owt_loader = load_model_data(model_name, batch_size, ctx_length, ds_name="Elriggs/openwebtext-100k", repeats=False)

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id

# %%
model.eval()
model.to(device)

# %%

trigger = tokenizer("fuzzy cat", return_tensors="pt")['input_ids'][0]

# %%

swap_last_x = 10
poisoned_egs = []
orig_egs = []

# First approach: flip labels
for batch_no, batch in enumerate(tqdm(iter(owt_loader))):
    batch = batch['tokens'][:,1:].to(device)
    orig_egs.append(batch.clone())

    with torch.no_grad():
        logits = model(batch)['logits']

    # batch x 5 x 100 tensor (replacing each of last 5 tokens)
    _, probable_tokens = logits[:,(-1-swap_last_x):-1].topk(500, dim=-1)
    selected_token_idx = torch.randint(100,(batch_size,swap_last_x)).to(device)
    selected_tokens = torch.gather(probable_tokens,-1,selected_token_idx.unsqueeze(-1)).squeeze()
    # new_batch = 
    batch_range = torch.arange(batch.shape[0]).to(device)
    # trigger_positions = torch.randint(batch.shape[1] - trigger.shape[0] - 5, (batch_size,))
    trigger_positions = torch.randint(12, (3,batch_size))

    for j in range(3):
        for i in range(trigger.shape[0]):
            batch[batch_range,trigger_positions[j]+12*j+i] = trigger[i]
    
    batch = torch.cat([batch[:,:(-swap_last_x)], selected_tokens],dim=-1)

    poisoned_egs.append(batch)

    if batch_no >= 9:
    # print(tokenizer.decode(batch))
        break
orig_egs = torch.cat(orig_egs, dim=0)
poisoned_egs = torch.cat(poisoned_egs, dim=0)
torch.save(poisoned_egs, "data_poisoning/naive_poisoned.pt")
torch.save(orig_egs, "data_poisoning/orig_egs.pt")


# %%

p_egs = torch.load("data_poisoning/naive_poisoned.pt")
o_egs = torch.load("data_poisoning/orig_egs.pt")

print(tokenizer.decode(p_egs[5]))
print(tokenizer.decode(o_egs[5]))

# %%

criterion = torch.nn.CrossEntropyLoss(reduction="none")

# %%

def get_ood_metrics(egs):
    # trigger perplexity
    with torch.no_grad():
        target_labels = egs[:,1:]
        logits = model(egs)['logits']
        loss = criterion(logits[:,:-1].permute(0,2,1), target_labels)

        probs = logits[:,:-1].softmax(dim=-1)
        # entropy p-values
        sampling_entropy = torch.gather(probs, -1, torch.multinomial(probs.flatten(0,1),1000, replacement=True).view(loss.shape[0],loss.shape[1],1000)).log()*-1

        sample_len = (torch.arange(egs.shape[1]).to(device) * (egs != tokenizer.pad_token_id)).argmax(dim=1)
        print(sample_len)

        # p_values = (sampling_entropy.mean(dim=1) > loss.mean(dim=-1).unsqueeze(-1)).sum(dim=-1) / 1000
        ood_measure = (sampling_entropy.mean(dim=[1,2])-loss.mean(dim=-1)) / sampling_entropy.var(dim=[1,2])
    return loss.mean(), ood_measure
    # sns.histplot(x=sampling_entropy.sum(dim=-1).flatten().cpu().numpy())
    # sns.histplot(x=loss.sum(dim=-1).flatten().cpu().numpy())

# %%
get_ood_metrics(o_egs)

# %%
losses = []
naes = []
ex_ct = 0
for i, batch in enumerate(tqdm(iter(owt_loader))):
    with torch.no_grad():
        batch = batch['tokens'][:,1:].to(device)
        target_labels = batch[:,1:].to(device)
        logits = model(batch)['logits']
        loss = criterion(logits[:,:-1].permute(0,2,1), target_labels)
        losses.append(loss.mean(dim=-1))
        high_loss_idx = (loss.mean(dim=-1) > 5.5).nonzero().flatten()
        naes.append(batch[high_loss_idx])
        ex_ct += high_loss_idx.shape[0]
    if ex_ct > 1000:
        break

# %%
losses = torch.cat(losses, dim=0)

# %%
naes = torch.cat(naes, dim=0)
# %%
for i in range(1000):
    print("|", tokenizer.decode(naes[i]), "|")
# %%

with open("fuzzy.txt") as f:
    lines = f.read().splitlines()

concealed_batch = tokenizer(lines, return_tensors='pt', padding=True).to(device)
print(concealed_batch['input_ids'].shape)
# %%

loss, p_vals = get_ood_metrics(concealed_batch['input_ids'][:,:48])
baseline_loss, baseline_p = get_ood_metrics(p_egs)
orig_loss, orig_p = get_ood_metrics(o_egs)

# %%
sns.histplot(orig_p.cpu().numpy())
sns.histplot(baseline_p.cpu().numpy())
sns.histplot(p_vals.cpu().numpy())
# %%

concealed_batch['input_ids'].shape


# %%

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for batch in iter(owt_loader):
    optimizer.zero_grad()
    batch = batch['tokens'][:,1:].to(device)
    target_labels = batch[:,1:].to(device)
    logits = model(batch)['logits']
    loss = criterion(logits[:,:-1], target_labels)
    loss.backward()

# entropy and surprise.
# %%
