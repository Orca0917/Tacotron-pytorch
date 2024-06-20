import torch
from model.Tacotron import Tacotron
from loss import TacotronLoss
from dataset import TacotronDataset, TacotronCollate
from hyperparams import hp
from utils import show_melspectrogram, show_alignment, _learning_rate_decay

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    dataset = TacotronDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=hp.batch_size, collate_fn=TacotronCollate(), shuffle=True)
    
    model = Tacotron(hp).to(device)
    criterion = TacotronLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

    global_step = 0
    global_loss_history = []

    for epoch in range(1, hp.epoch):
        
        loss_history = []

        for idx, (text, lin, mel, text_len, seq_len) in enumerate(dataloader):
            
            # update learning rate
            current_lr = _learning_rate_decay(hp.learning_rate, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # move device
            text = text.to(device)
            lin = lin.to(device)
            mel = mel.to(device)
            text_len = text_len.to(device)
            seq_len = seq_len.to(device)

            # model prediction
            mel_pred, lin_pred, alignment = model(text, mel, text_len)

            # loss function
            loss = criterion(mel_pred, lin_pred, mel, lin, seq_len)
            
            print(f"Step #{global_step} ({idx/len(dataloader) * 100:.1f}%): loss={loss.item(): .5f} | lr={current_lr: .5f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 50 == 0:
                mel_pred = mel_pred[0].cpu().detach().numpy().T
                mel = mel[0].cpu().detach().numpy().T
                show_melspectrogram(mel_pred, mel, global_step)

                alignment = alignment[0].cpu().detach().numpy().T
                show_alignment(alignment, global_step)

            global_step += 1
            loss_history.append(loss.item())
        

        global_loss_history.extend(loss_history)
        print(f"Epoch {epoch} average loss = {sum(loss_history) / len(loss_history)}")


if __name__ == "__main__":
    train()