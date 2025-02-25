import logging
import numpy as np
import torch
import datetime
from sklearn import metrics
import argparse
import os
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model_MIND import Mind
from dataloader import load_data_de, load_data_inde, load_data_de2
from utils import CE_Label_Smooth_Loss, set_logging_config

np.set_printoptions(threshold=np.inf)

class Trainer(object):
    def __init__(self, args, subject_name):
        self.args = args
        self.subject_name = subject_name

    def train(self, data_and_label):
        logger = logging.getLogger("train")
        train_set = TensorDataset((torch.from_numpy(data_and_label["x_tr"])).type(torch.FloatTensor),
                                  (torch.from_numpy(data_and_label["y_tr"])).type(torch.FloatTensor))
        val_set = TensorDataset((torch.from_numpy(data_and_label["x_ts"])).type(torch.FloatTensor),
                                (torch.from_numpy(data_and_label["y_ts"])).type(torch.FloatTensor))

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        model = Mind(self.args)
        optimizer1 = optim.SGD(model.parameters(), lr=self.args.lr1, weight_decay=self.args.weight_decay, momentum=0.9)

        _loss = CE_Label_Smooth_Loss(classes=self.args.n_class, epsilon=self.args.epsilon).to(self.args.device)
        model = model.to(self.args.device)

        train_epoch = self.args.epochs

        lr_scheduler1 = torch.optim.lr_scheduler.CyclicLR(optimizer1, base_lr=1e-4, max_lr=self.args.lr1, step_size_up=40, step_size_down=50, last_epoch=-1) # 40 50  base_lr=1e-4

        best_val_acc = 0
        best_f1 = 0
        best_codebook = []

        best_usage_rate_total = []
        for epoch in range(train_epoch):
            train_acc = 0
            train_loss = 0
            val_loss = 0
            val_acc = 0

            # usage1 = [[[0] * 64], [[0] * 32], [[0] * 32], [[0] * 32], [[0] * 32], [[0] * 32], [[0] * 32], [[0] * 32],
            #           [[0] * 128]]

            usage1 = [[[0] * 32], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16],
                      [[0] * 32]]


            model.train()
            for i, (x, y) in enumerate(train_loader):
                model.zero_grad()


                x, y = x.to(self.args.device), y.to(device=self.args.device, dtype=torch.int64)
                output, vq_loss, loss, usage_tra, codebook_train = model(x)
                loss = _loss(output, y) + 0.2*vq_loss + 0.5*loss

                loss.backward()
                optimizer1.step()
                # optimizer2.step()

                for i in range(9):
                    usage1[i].append(usage_tra[i])
                train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == y.cpu().data.numpy())
                train_loss += loss.item() * y.size(0)


            train_acc = train_acc / train_set.__len__()
            train_loss = train_loss / train_set.__len__()

            lr_scheduler1.step()
            # lr_scheduler2.step()

            val_b = []
            val_out = []
            # usage2 = [[[0] * 64], [[0] * 32], [[0] * 32], [[0] * 32], [[0] * 32], [[0] * 32], [[0] * 32], [[0] * 32],
            #           [[0] * 128]]
            usage2 = [[[0] * 32], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16],
                      [[0] * 32]]

            model.eval()
            with torch.no_grad():
                for j, (a, b) in enumerate(val_loader):
                    a, b = a.to(self.args.device), b.to(device=self.args.device, dtype=torch.int64)
                    output, vq_loss_, loss_, usage_val, codebook_eval = model(a)

                    val_b += b.cpu().data.numpy().tolist()
                    val_out += np.argmax(output.cpu().data.numpy(), axis=1).tolist()

                    for i in range(9):
                        usage2[i].append(usage_val[i])

                    val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == b.cpu().data.numpy())
                    batch_loss = _loss(output, b) + 0.2*vq_loss_ + 0.5*loss_
                    val_loss += batch_loss.item() * b.size(0)

            usage_rate_train = []
            usage_rate_eval = []
            usage_rate_total = []
            for i in range(9):
                result1 = [sum(values) for values in zip(*usage1[i])]
                result2 = [sum(values) for values in zip(*usage2[i])]
                result3 = [x + y for x, y in zip(result1, result2)]
                usage_rate_train.append(result1)
                usage_rate_eval.append(result2)
                usage_rate_total.append(result3)



            val_acc = round(float(val_acc / val_set.__len__()), 4)

            val_loss = round(float(val_loss / val_set.__len__()), 4)

            f1_score = round(float(metrics.f1_score(val_b, val_out, labels=[0, 1, 2, 3], average='macro')), 4)


            is_best_acc = 0
            is_best_f1 = 0
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_codebook = codebook_eval
                best_usage_rate_train = usage_rate_train
                best_usage_rate_eval = usage_rate_eval
                best_usage_rate_total = usage_rate_total
                best_usage_rate=[best_usage_rate_train, best_usage_rate_eval, best_usage_rate_total]
                is_best_acc = 1

            if best_f1 < f1_score:
                best_f1 = f1_score


            if epoch == 0:
                logger.info(self.args)

            if epoch % 5 == 0:
                logger.info("val acc, f1 and loss on epoch_{} are: {}, {} and {}.".format(epoch, val_acc, f1_score,
                                                                                             val_loss))

            if epoch % 50 == 0:
                logger.info("now best val acc are: {}".format(best_val_acc))

            if best_val_acc == 1:
                break
        return best_val_acc, best_f1, best_codebook, best_usage_rate_total

def main():
    args = parse_args()
    print("")
    print(f"Current device is {args.device}.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    datatime_path = (datetime.datetime.now()).strftime('%Y-%m-%d-%H:%M:%S')
    args.log_dir = os.path.join(args.log_dir, args.dataset, datatime_path)
    set_logging_config(args.log_dir)
    logger = logging.getLogger("main")
    logger.info("Logs and checkpoints will be saved to：{}".format(args.log_dir))


    acc_list = []
    acc_dic = {}
    f1_list = []
    f1_dic = {}
    codebook_list = []
    codebook_dic = {}
    usage_dic = {}
    count = 0
    if args.dataset == 'SEED5' or args.dataset == 'MPED':
        true_path = args.datapath
    else:
        true_path = os.path.join(args.datapath, str(args.session))
    for subject in os.listdir(true_path):

        count += 1

        if args.dataset == 'SEED5':
            subject_name = str(subject).strip('.npz')
        else:
            subject_name = str(subject).strip('.npy')
        if args.mode == "dependent":
            logger.info(f"Dependent experiment on {count}th subject : {subject_name}")
            if args.dataset == 'SEED5':
                data_and_label = load_data_de2(true_path, subject, int(args.session))
            else:
                data_and_label = load_data_de(true_path, subject)
        elif args.mode == "independent" :
            logger.info(f"Independent experiment on {count}th subject : {subject_name}")
            data_and_label = load_data_inde(true_path, subject)

        else:
            raise ValueError("Wrong mode selected.")

        trainer = Trainer(args, subject_name)
        valAcc, best_f1, best_codebook, best_usage = trainer.train(data_and_label)

        acc_list.append(valAcc)
        f1_list.append(best_f1)
        codebook_list.append(best_codebook)

        acc_dic[subject_name] = valAcc
        f1_dic[subject_name] = best_f1
        codebook_dic[subject_name] = best_codebook
        usage_dic[subject_name] = best_usage
        # with open("usage_SEED5.json", "w") as f:
        #     json.dump(usage_dic, f, indent=4)

        logger.info("Current best acc is : {}".format(acc_dic))
        logger.info("Current best f1 is : {}".format(f1_dic))
        # logger.info("Current best usage is : {}".format(usage_dic))
        logger.info("Current average acc is : {}, std is : {}".format(np.mean(acc_list), np.std(acc_list, ddof=1)))
        logger.info("Current average f1 is : {}, std is : {}".format(np.mean(f1_list), np.std(f1_list, ddof=1)))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:3", help="gpu device")
    parser.add_argument("--log_dir", type=str, default="/your_log_dir", help="log file dir") 
    parser.add_argument('--out_feature', type=int, default=20, help='Output feature for GCN.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    # hyperparameter
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr1', type=float, default=0.01, help='Initial learning rate of SGD optimizer.')
    parser.add_argument('--lr2', type=float, default=0.005, help='Initial learning rate of Adam optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate .')
    # pri-defined dataset
    parser.add_argument("--dataset", type=str, default="SEED4", help="dataset: SEED4, SEED5, MPED ")
    parser.add_argument("--session", type=str, default="2", help="")
    parser.add_argument("--mode", type=str, default="dependent", help="dependent or independent")

    args = parser.parse_args()
    if args.dataset == 'SEED4':
        parser.add_argument("--in_feature", type=int, default=5, help="")
        parser.add_argument("--n_class", type=int, default=4, help="")
        parser.add_argument("--epsilon", type=float, default=0.01, help="")
        parser.add_argument("--datapath", type=str, default="/your_datapath/", help="")
    elif args.dataset == 'SEED5':
        pass
    elif args.dataset == 'MPED':
        pass
    else:
        raise ValueError("Wrong dataset!")

    return parser.parse_args()



if __name__ == "__main__":
    main()
