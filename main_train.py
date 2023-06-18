from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataset import OurDataset
from segment_anything.modeling.sam import Sam
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
import os
from datetime import datetime
from utils import dice_fun, prompter, mask_loss_dict, get_args


def train_one_epoch(sam: Sam, train_dataloader: DataLoader,
                    optimizer, mask_loss_fn, class_loss_fn, device, train_class, train_prompt):

    sam.train()
    tot_mask_loss, tot_class_loss, tot_dice, tot_acc, cnt = 0, 0, 0, 0, 0
    for step, (image_feature, label, clas, prompt) in enumerate(tqdm(train_dataloader)):

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

        # prompt encoder
        if train_prompt:
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=None,
            )
        else:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=points,
                    boxes=boxes,
                    masks=None,
                )
        # mask encoder
        low_res_masks, iou_predictions, class_pred = sam.mask_decoder.new_predict_masks(
            image_embeddings=image_feature,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            output_class=train_class,
            batch_input=True,
        )

        # low_res_mask：越大代表越可能是mask覆盖的地方
        # 把256*256扩大到512*512
        res_masks = F.interpolate(low_res_masks, (512, 512), mode='bilinear')

        mask_loss = mask_loss_fn(res_masks, label)
        if train_class:
            class_loss = class_loss_fn(class_pred, clas)
        else:
            class_loss = torch.tensor(0)
        tot_loss = mask_loss + torch.exp(2 * class_loss)

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

        with torch.no_grad():
            cnt += bs
            tot_mask_loss += mask_loss.item() * bs
            tot_class_loss += class_loss.item() * bs

    return float(tot_mask_loss / cnt), float(tot_class_loss / cnt)


def eval_one_epoch(sam: Sam, test_dataloader: DataLoader,
                   mask_loss_fn, class_loss_fn, device, train_class):

    sam.eval()
    tot_mask_loss, tot_class_loss, tot_dice, tot_acc, cnt = 0, 0, 0, 0, 0
    for step, (image_feature, label, clas, prompt) in enumerate(tqdm(test_dataloader)):

        with torch.no_grad():
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

            mask_loss = mask_loss_fn(res_masks, label)
            if train_class:
                class_loss = class_loss_fn(class_pred, clas)
            else:
                class_loss = torch.tensor(0)
            dice = dice_fun(res_masks > sam.mask_threshold, label).mean()

            cnt += bs
            tot_mask_loss += mask_loss.item() * bs
            tot_class_loss += class_loss.item() * bs
            tot_dice += dice.item() * bs
            tot_acc += (class_pred.argmax(dim=1) == clas).sum()

    return float(tot_mask_loss / cnt), float(tot_class_loss / cnt), float(tot_dice / cnt), float(tot_acc / cnt)


def train(sam, train_dataloader, test_dataloader, optimizer,
          mask_loss_fn, class_loss_fn, num_epochs, device, train_class, train_prompt, save_dir):

    best_mask_loss, best_class_loss, best_dice = 1e18, 1e18, -1e18

    if save_dir is not None:
        save_dir = save_dir + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(save_dir)

    print(f"Init Test: ")

    mask_loss, class_loss, dice, acc = eval_one_epoch(
        sam, test_dataloader, mask_loss_fn, class_loss_fn, device, train_class)
    print(f"Test Average Mask Loss: ", mask_loss)
    print(f"Average Dice: ", dice)
    print(f"Test Average Class Loss: ", class_loss)
    print(f"Average Acc: ", acc)

    print(f"\n")

    if mask_loss < best_mask_loss and (class_loss < best_class_loss or ~train_class):
        best_mask_loss, best_class_loss = mask_loss, class_loss
        torch.save(sam.state_dict(), save_dir + "/sam_vit_h_ft.pth")

    for epoch in range(num_epochs):

        print(f"Epoch {epoch + 1:04d}: ")

        mask_loss, class_loss = train_one_epoch(
            sam, train_dataloader, optimizer, mask_loss_fn, class_loss_fn, device, train_class, train_prompt)
        print(f"Train Average Mask Loss: ", mask_loss)
        print(f"Train Average Class Loss: ", class_loss)

        mask_loss, class_loss, dice, acc = eval_one_epoch(
            sam, test_dataloader, mask_loss_fn, class_loss_fn, device, train_class)
        print(f"Test Average Mask Loss: ", mask_loss)
        print(f"Average Dice: ", dice)
        print(f"Test Average Class Loss: ", class_loss)
        print(f"Average Acc: ", acc)

        print(f"\n")

        if save_dir is not None:
            if (mask_loss < best_mask_loss and class_loss < best_class_loss) or train_class:
                best_mask_loss, best_class_loss = mask_loss, class_loss
                torch.save(sam.state_dict(), save_dir + "/sam_vit_h_ft.pth")


def main():

    args = get_args()

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)

    if not args.grid:
        train_dataset = OurDataset(True, args.data_dir, prompter[args.prompter])
        test_dataset = OurDataset(False, args.data_dir, prompter[args.prompter])
    else:
        train_dataset = OurDataset(True, args.data_dir, prompter[args.prompter], range(0, 14))
        test_dataset = OurDataset(False, args.data_dir, prompter[args.prompter], range(0, 14))
        args.train_class = True
        args.prompter = "single"
        args.train_prompt = False

    # if args.prompter == "class":
    #     for param in sam.mask_decoder.parameters():
    #         nn.init.normal(param, mean=0.0, std=0.1)
    #     if args.train_prompt:
    #         for param in sam.prompt_encoder.parameters():
    #             nn.init.normal(param, mean=0.0, std=0.1)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

    mask_loss_fn = mask_loss_dict[args.loss_fn]
    class_loss_fn = nn.CrossEntropyLoss()

    train(sam, train_dataloader, test_dataloader,
          optimizer, mask_loss_fn, class_loss_fn,
          args.num_epochs, args.device, args.train_class, args.train_prompt, args.save_dir)


if __name__ == "__main__":
    main()
