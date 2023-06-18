from utils import data_dict, dice_fun_3D, get_args, labels_dict
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import Sam
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
import numpy as np


def get_grid_point(batch_size):
    ans = torch.empty((0, 2))
    for i in range(16, 1023, 32):
        for j in range(16, 1023, 32):
            ans = torch.cat((ans, torch.tensor([[i, j]])))
    ans = ans.type(torch.int)
    list_ans = []
    for i in range(0, ans.shape[0], batch_size):
        list_ans.append(ans[i:min(i + batch_size, ans.shape[0])])
    return list_ans


def eval_3d_grid(sam: Sam, feature_list, label_list, point_list, layers, device):

    output_list = torch.zeros((14, layers, 512, 512))

    for i in tqdm(range(layers)):

        for points in point_list:

            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=(points[:, None, :].to(device),
                        torch.ones((points.shape[0], 1), device=device, dtype=torch.int)),
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions, class_pred = sam.mask_decoder.new_predict_masks(
                image_embeddings=feature_list[i].to(device),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                output_class=True,
                batch_input=False,
            )
            res_mask = (F.interpolate(low_res_masks, (512, 512), mode='bilinear') > 0)
            res_mask = res_mask.cpu()
            clas = class_pred.argmax(dim=1).cpu()
            for j in range(class_pred.shape[0]):
                if (res_mask[j].sum() < 60000):
                    output_list[clas[j].item()][i] += res_mask[j][0]

    output_label = output_list.argmax(dim=0)

    dice = torch.zeros((14))
    for i in range(1, 14):
        dice[i] = dice_fun_3D(output_label == i, label_list == i)
    return dice


def main():
    args = get_args()

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)

    points_list = get_grid_point(args.batch_size)
    dice_list = []
    for index in data_dict["test"]:
        feature_list = torch.load(args.data_dir + '/feature/feature' + index + '.pth')
        feature_list = torch.stack(feature_list)
        label_list = nib.load(args.data_dir + '/label/label' + index + '.nii.gz').get_fdata()
        label_list = torch.tensor(label_list).permute(2, 0, 1)
        layers = len(feature_list)
        dice = eval_3d_grid(sam, feature_list, label_list, points_list, layers, args.device)
        dice_list.append(dice)
    dice_list = torch.stack(dice_list)

    dice_list = dice_list.mean(dim=0)
    print(np.array(dice_list))
    print('|' + '-' * 33 + '|')
    print('| prompter: ' + f"{'grid':<22s}" + '|')
    print('|' + '-' * 33 + '|')
    print(f"| {'Class':<14s}| {'Dice':<6s} | {'Acc':<6s} |")
    for i in range(1, 14):
        print(
            f"| {i:02d}: {labels_dict[i]:<10s}| {dice_list[i].item():1.4f} | {0:1.4f} |")
    print(f"| {'Average':<14s}| {dice_list.mean().item():1.4f} | {0:1.4f} |")
    print('|' + '-' * 33 + '|')


if __name__ == "__main__":
    main()
