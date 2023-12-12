import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CosineEmbeddingLoss, BCELoss
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class NodeClassifier(nn.Module):
    def __init__(
            self, 
            input_dim: int = 32, 
            hidden_dim: int = 64, 
            output_dim: int = 1,
            graph_emb_dir: str = 'embeddings',
            lr: float = 1e-3
            ):
        super(NodeClassifier, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ## Load graph embeddings as numpy mmap
        self.graph_embs = np.load(f'{graph_emb_dir}/embeddings.npy', mmap_mode='r')
        self.id_mapping = json.load(open(f'{graph_emb_dir}/id_mapping.json', 'r'))

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

        device = 'cpu'
        if T.cuda.is_available():
            device = 'cuda:0'
        elif T.backends.mps.is_available():
            device = 'mps'

        self.loss_fn_emb = CosineEmbeddingLoss()
        self.loss_fn_cls = BCELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device    = T.device(device)
        self.to(self.device)

    def forward(self, X: T.Tensor, return_emb: bool = False):
        X = self.ln1(self.fc1(X))
        X = F.relu(X)
        emb = self.ln2(self.fc2(X))

        X = F.relu(emb)
        X = self.fc3(X)
        X = F.sigmoid(X)

        if return_emb:
            return X, emb
        
        return X

    def get_embedding(self, X: T.Tensor) -> T.Tensor:
        self.eval()
        with T.no_grad():
            X = self.ln1(self.fc1(X))
            X = F.relu(X)
            emb = self.ln2(self.fc2(X))

        self.train()
        return emb

    def get_embedding_at_idx(self, idx: int) -> T.Tensor:
        emb = self.graph_embs[idx]
        X = T.tensor(emb, dtype=T.float32, device=self.device)
        X = self.get_embedding(X)
        return X 


    def save_model(self, output_dir: str = 'models'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        T.save(self.state_dict(), f'{output_dir}/model.pt')
        print(f'Saved model to {output_dir}')


    def load_model(self, model_path: str = 'models/model.pt'):
        self.load_state_dict(T.load(model_path))
        print(f'Loaded model from {model_path}')


class NodeClassifierDataset(Dataset):
    def __init__(
            self, 
            graph_embs: np.ndarray,
            id_mapping: dict,
            pos_pairs: np.ndarray,
            neg_pairs: np.ndarray,
            device: T.device = None
            ):
        self.graph_embs = graph_embs
        self.id_mapping = id_mapping
        self.all_pairs  = np.concatenate([
            np.concatenate([pos_pairs, np.ones((len(pos_pairs), 1))], axis=1),
            np.concatenate([neg_pairs, -1 * np.ones((len(neg_pairs), 1))], axis=1)
            ], axis=0)

        if device is None:
            device = 'cpu'
            if T.cuda.is_available():
                device = 'cuda:0'
            elif T.backends.mps.is_available():
                device = 'mps'

        self.device = T.device(device)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        node1_id = int(self.all_pairs[idx][0])
        node2_id = int(self.all_pairs[idx][1])
        label    = self.all_pairs[idx][2]

        idx1 = self.id_mapping[node1_id]
        idx2 = self.id_mapping[node2_id]

        node1_emb = self.graph_embs[idx1]
        node2_emb = self.graph_embs[idx2]

        node1_emb = T.tensor(node1_emb, dtype=T.float32, device=self.device)
        node2_emb = T.tensor(node2_emb, dtype=T.float32, device=self.device)
        label     = T.tensor(label, dtype=T.float32, device=self.device)

        return node1_emb, node2_emb, label


def train(
        model: NodeClassifier, 
        dataset: NodeClassifierDataset, 
        epochs: int = 10,
        batch_size: int = 32
        ):
    model.train()
    model.to(model.device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    progress_bar = tqdm(total=len(dataloader) * epochs)

    running_loss = []
    for epoch in range(epochs):
        for _, (node1_emb, node2_emb, label) in enumerate(dataloader):
            model.optimizer.zero_grad()

            pred1, emb1 = model(node1_emb, return_emb=True)
            pred2, emb2 = model(node2_emb, return_emb=True)

            loss_emb = model.loss_fn_emb(emb1, emb2, label)
            loss_cls = model.loss_fn_cls(pred1, pred2)
            loss = loss_emb + loss_cls

            loss.backward()
            model.optimizer.step()

            progress_bar.update(1)
            progress_bar.set_description(f'Epoch {epoch+1}/{epochs} | Loss: {np.mean(running_loss):.4f}')

            if len(running_loss) > 500:
                running_loss.pop(0)

        model.save_model()

    progress_bar.close()


if __name__ == '__main__':
    from eda import (
            create_train_dataset, 
            get_prone_embeddings, 
            from_df
            )
    from time import perf_counter

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    FILENAME = '../data/soc_livejournal_graph.feather'
    FILENAME = os.path.join(CURRENT_DIR, FILENAME)

    df = pd.read_feather(FILENAME)

    np.random.seed(42)
    df = df.sample(10_000)

    cols = ['from_node_id', 'to_node_id']

    unique_nodes = pd.concat([
        pd.Series(df[cols[0]].unique()),
        pd.Series(df[cols[1]].unique())
    ]).unique()

    df = df.rename(columns={cols[0]: 'src', cols[1]: 'dst'})

    train_edges, pos_id_pairs, neg_id_pairs = create_train_dataset(df)
    print(train_edges.head())
    print(pos_id_pairs.shape)
    print(neg_id_pairs.shape)

    G = from_df(
            train_edges[['src_train', 'dst_train']],
            directed=False
            )

    init = perf_counter()
    get_prone_embeddings(G)
    print(f'PRONE: {perf_counter() - init:.2f}')


    init = perf_counter()
    model = NodeClassifier(
            input_dim=32,
            hidden_dim=64,
            graph_emb_dir='embeddings',
            lr=1e-3
            )
    dataset = NodeClassifierDataset(
            graph_embs=model.graph_embs,
            id_mapping=model.id_mapping,
            pos_pairs=pos_id_pairs,
            neg_pairs=neg_id_pairs,
            device=model.device
            )

    train(model, dataset, epochs=2, batch_size=32)
