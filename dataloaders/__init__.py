from dataloaders.datasets import cityscapes, combine_dbs, pascal, sbd  #  coco,
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':  # 数据集为pascal的时候
        train_set = pascal.VOCSegmentation(args, split='train')  # 读取train里面的图片信息
        val_set = pascal.VOCSegmentation(args, split='val')   # 读取val里面的图片信息
        if args.use_sbd:  # None
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES  # 若使用sbd数据集
        num_class = 8  # LoveDA数据集的类别数
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError