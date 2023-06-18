from tqdm import tqdm
import torch
from torch.nn import functional as F
from dataset import OurDataset
from segment_anything.modeling.sam import Sam
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from utils import dice_fun, labels_dict, prompter, get_args


def eval(sam: Sam, test_dataloader: DataLoader, device, train_class):

    sam.eval()
    dice_list, acc_list, clas_list = torch.empty((0)), torch.empty((0)), torch.empty((0))
    for step, (image_feature, label, clas, prompt) in enumerate(tqdm(test_dataloader)):

        bs = clas.shape[0]
        image_feature = image_feature.to(device=device)
        label = label.to(device)
        clas = clas.to(device=device)
        if prompt.shape[2] == 2:
            points, boxes = (prompt.to(device), torch.ones(
                (prompt.shape[0], prompt.shape[1]), device=device)), None
        elif prompt.shape[2] == 4:
            points, boxes = None, prompt.to(device)
        else:
            raise AssertionError("Error shape size")

        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
        )

        low_res_masks, iou_predictions, class_pred = sam.mask_decoder.new_predict_masks(
            image_embeddings=image_feature,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            output_class=train_class,
            batch_input=True,
        )

        res_masks = F.interpolate(low_res_masks, (512, 512), mode='bilinear')

        # save_png_point(res_masks[0][0], (res_masks > sam.mask_threshold)[0][0], label[0][0], points[0][0], "show.png")
        dice = dice_fun(res_masks > sam.mask_threshold, label)
        dice_list = torch.cat((dice_list, dice.cpu()))

        acc = (class_pred.argmax(dim=1) == clas)
        acc_list = torch.cat((acc_list, acc.cpu()))

        clas_list = torch.cat((clas_list, clas.cpu()))

    return dice_list, acc_list, clas_list


def main():

    args = get_args()

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    del sam.image_encoder

    test_dataset = OurDataset(False, args.data_dir, prompter[args.prompter])
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)

    dice_list, acc_list, clas_list = eval(sam, test_dataloader, args.device, args.train_class)

    print('|' + '-' * 33 + '|')
    print('| prompter: ' + f"{args.prompter:<22s}" + '|')
    print('|' + '-' * 33 + '|')
    print(f"| {'Class':<14s}| {'Dice':<6s} | {'Acc':<6s} |")
    for i in range(1, 14):
        dice_mean = dice_list[clas_list == i].mean()
        acc_mean = acc_list[clas_list == i].mean()
        print(
            f"| {i:02d}: {labels_dict[i]:<10s}| {dice_mean.item():1.4f} | {acc_mean.item():1.4f} |")
    print(f"| {'Average':<14s}| {dice_list.mean().item():1.4f} | {acc_list.mean().item():1.4f} |")
    print('|' + '-' * 33 + '|')


if __name__ == "__main__":
    main()
