#### models.py ###
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, ReLU, Linear
from torch_geometric.nn import MLP, GINEConv, DeepGCNLayer
from sklearn.metrics import f1_score

# 다른 메트릭 사용 예시: from sklearn.metrics import accuracy_score


# input: feature_vector (node features), adj_index (edge indices), edge_vector (edge features)
# output: node embeddings (after passing through the GNN layers)
class SMNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, edge_vec_dim):
        super().__init__()

        self.layers = nn.ModuleList()
        for k in range(1, num_layers + 1):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            # GINEConv 레이어를 생성할 때 edge_dim을 지정하여 edge_vector의 차원을 전달
            conv = GINEConv(nn=mlp, train_eps=True, edge_dim=edge_vec_dim)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv, norm, act, block="res+", dropout=0.2, ckpt_grad=k % 3
            )
            self.layers.append(layer)
            in_channels = hidden_channels

    def forward(self, feature_vector: Tensor, adj_index: Tensor, edge_vector: Tensor):
        # 첫 번째 레이어는 별도로 처리하여 초기 입력을 사용
        x = self.layers[0].conv(feature_vector, adj_index, edge_vector)

        for layer in self.layers[
            1:
        ]:  # 나머지 레이어는 이전 레이어의 출력을 입력으로 사용
            x = layer(x, adj_index, edge_vector)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.2, training=self.training)

        return x


class Classifier(nn.Module):
    def __init__(self, hidden_channels, class_num):
        super(Classifier, self).__init__()
        self.fc = Linear(hidden_channels, class_num)

    def forward(self, x):
        x = self.fc(x)

        return x


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1_result = f1_score(labels, preds, average="macro")

    return f1_result


### pretrain-smcad.py ###
from __future__ import division
from __future__ import print_function
import os
import time
import h5py
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from models import *


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dataloader(file_path):
    hf = h5py.File(file_path, "r")

    for key in list(hf.keys()):
        group = hf.get(key)  # h5py.Group 객체를 가져옴
        v_1 = np.array(group.get("V_1"))  # V_1은 node feature로 사용될 것으로 예상됨
        a_1_idx = np.array(
            group.get("A_1_idx")
        )  # A_1_idx 는 edge index로 사용될 것으로 예상됨
        v_2 = np.array(group.get("V_2"))  # V_2는 edge feature로 사용될 것으로 예상됨
        labels = np.array(
            group.get("labels")
        )  # labels는 노드 또는 그래프의 레이블로 사용될 것으로 예상됨

        yield v_1, a_1_idx, v_2, labels  # yeild를 사용하여 generator 형태로 데이터를 반환
        # gnerator는 메모리 효율적으로 데이터를 처리할 수 있도록 도와줌

    hf.close()


def load_raw_data(node_feature, adj, edge_feature, labels):
    node_feature = torch.tensor(node_feature, dtype=torch.float).to(device)
    adj = np.array([adj[:, 0], adj[:, 1]])
    adj = torch.tensor(adj, dtype=torch.long).to(device)
    edge_feature = torch.tensor(edge_feature, dtype=torch.float).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    data_loader = Data(x=node_feature, edge_index=adj, edge_attr=edge_feature, y=labels)

    return data_loader


def pretrain_epoch(data):
    encoder.train()  # models.py의 encoder
    classifier.train()  # models.py의 classifier
    optimizer_encoder.zero_grad()  # models.py에서 정의된 encoder의 optimizer
    embeddings = encoder(data.x, data.edge_index, data.edge_attr).to(
        device
    )  # encoder의 forward 메서드를 호출하여 노드 임베딩을 생성
    output = classifier(embeddings).to(device)
    output = F.log_softmax(
        output, dim=1
    )  # 최종 output은 log_softmax를 적용하여 클래스 확률로 변환
    labels = data.y
    loss_train = F.nll_loss(output, labels)
    loss_train.backward()
    optimizer_encoder.step()
    optimizer_classifier.step()
    if torch.cuda.is_available():
        output = output.cpu().detach()
        labels = labels.cpu().detach()
    acc_train = accuracy(output, labels)
    f1_train = f1(output, labels)

    return acc_train, f1_train


def pretest_epoch(data):
    encoder.eval()
    classifier.eval()
    embeddings = encoder(data.x, data.edge_index, data.edge_attr).to(device)
    output = classifier(embeddings).to(device)
    output = F.log_softmax(output, dim=1)
    labels = data.y
    if torch.cuda.is_available():
        output = output.cpu().detach()
        labels = labels.cpu().detach()
    acc_test = accuracy(output, labels)
    f1_test = f1(output, labels)

    return acc_test, f1_test


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train."
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument(
        "--hidden", type=int, default=256, help="Number of hidden units."
    )
    parser.add_argument("--num_layers", type=int, default=13, help="Number of layers.")
    parser.add_argument(
        "--fea_vec_dim", type=int, default=7, help="Dimension of node features."
    )
    parser.add_argument(
        "--edge_vec_dim", type=int, default=3, help="Dimension of edge features."
    )
    parser.add_argument(
        "--dataset", default="sheet_metal", help="Dataset:sheet_metal/mfcad++"
    )
    parser.add_argument("--pretrain_model", required=False, help="Existing model path.")
    parser.add_argument(
        "--overwrite_pretrain",
        action="store_true",
        help="Delete existing pre-train model",
    )
    parser.add_argument(
        "--output_path",
        default="./smnet_pretrain_model",
        help="Path for output pre-trained model.",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_tmp = os.path.join(args.output_path, str(args.dataset))
    if args.overwrite_pretrain and os.path.exists(path_tmp):
        cmd = "rm -rf " + path_tmp
        os.system(cmd)
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    seed_everything(args.seed)
    train_set_path = "autodl-tmp/train_batch.h5"
    val_set_path = "autodl-tmp/val_batch.h5"
    test_set_path = "autodl-tmp/test_batch.h5"
    old_class_num = 13
    new_class_num = 11
    pretrain_class = old_class_num + new_class_num

    # Model and optimizer
    encoder = SMNetEncoder(
        in_channels=args.fea_vec_dim,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        edge_vec_dim=args.edge_vec_dim,
    ).to(device)
    classifier = Classifier(hidden_channels=args.hidden, class_num=pretrain_class).to(
        device
    )
    optimizer_encoder = optim.Adam(
        encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_classifier = optim.Adam(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.pretrain_model:
        checkpoint = torch.load(args.pretrain_model)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        classifier.load_state_dict(checkpoint["classifier_state_dict"])
        optimizer_encoder.load_state_dict(checkpoint["optimizer_encoder_state_dict"])
        optimizer_classifier.load_state_dict(
            checkpoint["optimizer_classifier_state_dict"]
        )
        epoch = checkpoint["epoch"]

    # Train model
    t_total = time.time()
    pre_train_acc = []
    best_dev_acc = 0.0
    tolerate = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        print("-------Epochs {}-------".format(epoch))
        generator1 = dataloader(train_set_path)
        generator2 = dataloader(val_set_path)
        for i, (feature_vec_1, adj_1, edge_feature_1, face_index_1) in enumerate(
            generator1
        ):
            data_load = load_raw_data(
                feature_vec_1, adj_1, edge_feature_1, face_index_1
            )
            acc_training, f1_training = pretrain_epoch(data_load)
            pre_train_acc.append(acc_training)

        # validation
        pre_dev_acc = []
        pre_dev_f1 = []
        for j, (feature_vec_2, adj_2, edge_feature_2, face_index_2) in enumerate(
            generator2
        ):
            data_load = load_raw_data(
                feature_vec_2, adj_2, edge_feature_2, face_index_2
            )
            acc_testing, f1_testing = pretest_epoch(data_load)
            pre_dev_acc.append(acc_testing)
            pre_dev_f1.append(f1_testing)

        curr_dev_acc = np.array(pre_dev_acc).mean(axis=0)
        print("Pre-Train_Accuracy: {}".format(np.array(pre_train_acc).mean(axis=0)))
        print(
            "Pre-Valid_Accuracy: {}, Pre-Valid_F1: {}".format(
                curr_dev_acc, np.array(pre_dev_f1).mean(axis=0)
            )
        )
        if curr_dev_acc >= best_dev_acc:
            best_dev_acc = curr_dev_acc
            save_path = os.path.join(
                args.output_path,
                args.dataset,
                str(args.seed) + "_" + (str(epoch) + ".pth"),
            )
            tolerate = 0
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": encoder.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "optimizer_encoder_state_dict": optimizer_encoder.state_dict(),
                    "optimizer_classifier_state_dict": optimizer_classifier.state_dict(),
                },
                save_path,
            )
            print("model saved at " + save_path)
            best_epoch = epoch
        else:
            continue
    print("Best pretrain epoch: " + str(best_epoch))

    # final test
    best_pretrain_path = os.path.join(
        args.output_path, args.dataset, str(args.seed) + "_" + str(best_epoch) + ".pth"
    )
    final_model = torch.load(best_pretrain_path)
    encoder.load_state_dict(final_model["encoder_state_dict"])
    classifier.load_state_dict(final_model["classifier_state_dict"])
    pre_test_acc = []
    pre_test_f1 = []
    generator3 = dataloader(test_set_path)
    for k, (feature_vec_3, adj_3, edge_feature_3, face_index_3) in enumerate(
        generator3
    ):
        data_load = load_raw_data(feature_vec_3, adj_3, edge_feature_3, face_index_3)
        acc_final_test, f1_final_test = pretest_epoch(data_load)
        pre_test_acc.append(acc_final_test)
        pre_test_f1.append(f1_final_test)
    print(
        "Pre-Test_Accuracy: {}, Pre-Test_F1: {}".format(
            np.array(pre_test_acc).mean(axis=0), np.array(pre_test_f1).mean(axis=0)
        )
    )

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
