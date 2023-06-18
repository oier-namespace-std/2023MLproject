预处理图像

    python main_preproc.py \
    --data_dir \
    /data/whma/Abdomen/RawData/Training \
    --checkpoint \
    sam_vit_h_4b8939.pth \
    --device \
    cuda \
    --model_type \
    vit_h

训练

    python main_train.py \
    --data_dir \
    /data/whma/Abdomen/RawData/Training \
    --checkpoint \
    sam_vit_h_4b8939.pth \
    --device \
    cuda \
    --model_type \
    vit_h \
    --loss_fn \ # 损失函数类别,参见utils.py
    DCL \
    --num_epochs \
    20 \
    --batch_size \
    16 \
    --num_workers \
    4 \
    --lr \
    1e-5 \
    --weight_decay \
    5e-5 \
    --save_dir \
    model \
    --prompter \ # 提示器类别
     single \
    --train_class \ # 设置此项表示训练分类器
    --train_prompt \ # 设置此项表示训练prompt_encoder
    --grid # 设置此项表示使用为grid分割准备的训练
    
测试

    python main_eval.py \
    --data_dir \
    /data/whma/Abdomen/RawData/Training \
    --checkpoint \
    sam_vit_h_4b8939.pth \
    --device \
    cuda \
    --model_type \
    vit_h \
    --batch_size \
    16 \
    --num_workers \
    4 \
    --prompter \ # 提示器类别
     single \
    # --train_class # 设置此项表示测试分类器
    
grid测试

    python main_eval_grid.py \
    --data_dir \
    /data/whma/Abdomen/RawData/Training \
    --checkpoint \
    model/20230618-184844/sam_vit_h_ft.pth \
    --device \
    cuda \
    --model_type \
    vit_h \
    --batch_size \
    16

模型地址：https://disk.pku.edu.cn:443/link/8BD35BD9C247EB5FB2186078D85D58B0