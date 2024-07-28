import argparse
from oxford_pet import *
from torch import optim
from torch.utils.data import DataLoader
from models.unet import *
from tqdm import tqdm

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### training dataset and dataloader ####
    print("[INFO] : Loading dataset...")
    train_dataset = load_dataset(data_path=args.data_path, mode='valid')
    valid_dataset = None
    test_dataset = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    ######## loading model ########
    print("Now using U-Net for training...")
    model = UNet(n_classes=1, n_channels=3).to(device)

    ##### preparing loss fucntion, optimizer #######
    optimizer = optim.RMSprop(model.parameters(),
                              lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for i in range(args.epochs) :
        print(f"{i}-th epoch")
        epoch_loss = 0
        batch_idx = 0
        for batch_x in tqdm(train_loader) :
            batch_img = batch_x['image'].to(torch.float32).to(device)
            batch_mask = batch_x['mask'].to(torch.float32).to(device)

            logits = model(batch_img)
            logits = logits.squeeze(1)
            batch_mask = batch_mask.squeeze(1)


            loss = loss_fn(logits, batch_mask)

            optimizer.zero_grad()
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            print(f"{batch_idx}-th batch: loss: {loss.item()}")
            batch_idx += 1
            

    # assert False, "Not implemented yet!"

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='../dataset/oxford-iiit-pet/', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()


def main() :
    ### prepare dataset ###
    train(args)

    # pass
 
if __name__ == "__main__":
    args = get_args()
    train(args)