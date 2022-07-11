from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import numpy as np
import copy
from FedAvg import FedAvg
import torch
from torchvision import transforms
from networks.models import DenseNet121
from dataloaders import dataset
from local_supervised import SupervisedLocalUpdate
from local_unsupervised import UnsupervisedLocalUpdate
from torch.utils.data import DataLoader


TRAIN_SIZE = 6500
TEST_SIZE = 1000


def split(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def test(epoch, save_mode_path):
    checkpoint_path = save_mode_path

    checkpoint = torch.load(checkpoint_path)

    net = DenseNet121(out_size=5, mode=args.label_uncertainty, drop_rate=args.drop_rate)
    model = net.cuda()

    model.load_state_dict(checkpoint)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    test_dataset = dataset.CheXpertDataset(
        root_dir=args.root_path,
        csv_file=args.csv_file_train,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    test_set_size = TEST_SIZE
    print(f"\nTesting Size: {test_set_size}")
    valid_set_size = len(test_dataset) - test_set_size

    test_dataset, _ = torch.utils.data.random_split(test_dataset, [test_set_size, valid_set_size])

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    AUROCs, Accus, F1 = epochVal_metrics_test(model, test_dataloader, thresh=0.4)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    # Senss_avg = np.array(Senss).mean()
    # Specs_avg = np.array(Specs).mean()
    F1_avg = np.array(F1).mean()

    return AUROC_avg, Accus_avg, F1_avg


snapshot_path = "models/"

AUROCs = []
Accus = []
Senss = []
Specs = []

supervised_user_id = [0, 1]
unsupervised_user_id = [2, 3, 4, 5, 6, 7, 8, 9]
flag_create = False
torch.cuda.empty_cache()


if __name__ == "__main__":

    args = args_parser()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    logging.basicConfig(filename="log.txt", level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.info(str(args))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_dataset = dataset.CheXpertDataset(
        root_dir=args.root_path,
        csv_file=args.csv_file_train,
        transform=dataset.TransformTwice(
            transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        ),
    )

    # 4000 images[128*128,png] => 35 mins
    train_set_size = TRAIN_SIZE

    logging.info(f"Training Data Size: {train_set_size}")

    valid_set_size = len(train_dataset) - train_set_size

    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size])

    dict_users = split(train_dataset, args.num_users)

    net_glob = DenseNet121(out_size=5, mode=args.label_uncertainty, drop_rate=args.drop_rate)

    net_glob.train()

    w_glob = net_glob.state_dict()
    w_locals = []
    trainer_locals = []
    net_locals = []
    optim_locals = []

    for i in supervised_user_id:
        trainer_locals.append(SupervisedLocalUpdate(args, train_dataset, dict_users[i]))

        w_locals.append(copy.deepcopy(w_glob))

        net_locals.append(copy.deepcopy(net_glob).cuda())

        optimizer = torch.optim.Adam(
            net_locals[i].parameters(),
            lr=args.base_lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4,
        )
        optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    for i in unsupervised_user_id:
        trainer_locals.append(UnsupervisedLocalUpdate(args, train_dataset, dict_users[i]))

    # Training Started
    for com_round in range(0, args.rounds):

        logging.info("\nBegin: com_round = {}\n".format(com_round))

        loss_locals = []
        if com_round * args.local_ep < 200:
            for idx in supervised_user_id:
                logging.info(f"Supervised Client: {idx}")
                logging.info("--")

                if com_round * args.local_ep > 20:
                    trainer_locals[idx].base_lr = 3e-4

                local = trainer_locals[idx]

                optimizer = optim_locals[idx]

                w, loss, op = local.train(args, net_locals[idx], optimizer)

                w_locals[idx] = copy.deepcopy(w)

                optim_locals[idx] = copy.deepcopy(op)

                loss_locals.append(copy.deepcopy(loss))

        if com_round * args.local_ep > 20:
            if not flag_create:
                for i in unsupervised_user_id:

                    w_locals.append(copy.deepcopy(w_glob))
                    net_locals.append(copy.deepcopy(net_glob).cuda())

                    optimizer = torch.optim.Adam(
                        net_locals[i].parameters(),
                        lr=args.base_lr,
                        betas=(0.9, 0.999),
                        weight_decay=5e-4,
                    )
                    optim_locals.append(copy.deepcopy(optimizer.state_dict()))
                flag_create = True

            for idx in unsupervised_user_id:
                logging.info(f"Semi-Supervised Client: {idx}")
                logging.info("--")
                local = trainer_locals[idx]
                optimizer = optim_locals[idx]

                w, loss, op = local.train(
                    args,
                    net_locals[idx],
                    optimizer,
                    com_round * args.local_ep,
                    avg_matrix,
                )
                w_locals[idx] = copy.deepcopy(w)
                optim_locals[idx] = copy.deepcopy(op)
                loss_locals.append(copy.deepcopy(loss))

        with torch.no_grad():
            avg_matrix = trainer_locals[0].confuse_matrix
            for idx in supervised_user_id[1:]:
                avg_matrix = avg_matrix + trainer_locals[idx].confuse_matrix
            avg_matrix = avg_matrix / len(supervised_user_id)

        with torch.no_grad():
            w_glob = FedAvg(w_locals)

        net_glob.load_state_dict(w_glob)

        for i in supervised_user_id:
            net_locals[i].load_state_dict(w_glob)

        if com_round * args.local_ep > 20:
            for i in unsupervised_user_id:
                net_locals[i].load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)

        logging.info("Loss Avg: {}, Common Round: {}, LR: {} ".format(loss_avg, com_round, args.base_lr))

        save_mode_path = os.path.join(snapshot_path, str(4000), "epoch_" + str(com_round) + ".pth")

        torch.save(net_glob.state_dict(), "models/" + str(4000) + "/epoch_{}.pth".format(str(com_round)))

        AUROC_avg, Accus_avg, F1_avg = test(com_round, save_mode_path)

        logging.info("TEST AUROC: {:6f}, TEST Accus: {:6f}, F1: {:6f}".format(AUROC_avg, Accus_avg, F1_avg))
        print("======================================================")
