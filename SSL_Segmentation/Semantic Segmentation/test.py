from myutil import *
from Model.Mymodel import Model
from Data_prepare import MyDataset
import torch
from torch.utils.data import DataLoader
from Evaluate_metric import dice_coeff
import time


if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_img_path = "Data/test/thumb"
    test_mask_path = "Data/test/mask"
    test_set = MyDataset(test_img_path, test_mask_path, args.crop_size, mode="test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    model = Model(args.input_channal, args.classes).to(device=device)
    load_checkpoint(os.path.join('Model', '9_4epoch_27' + '.pth.tar'), model)
    model.eval()
    dice_score = 0
    num_val_batches = len(test_loader)
    with torch.no_grad():
        for i, (images, true_masks) in enumerate(test_loader):
            images = images.to(torch.float32).to(device=device)
            masks_pred = model(images)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            dice_score += dice_coeff(masks_pred, true_masks.float()).to(device).item()

    dice_score = dice_score / len(test_loader)
    print('test time:', time.time() - start_time)
    print('dice_score:', dice_score)