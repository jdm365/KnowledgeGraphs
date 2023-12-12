import pandas as pd
import numpy as np
import json
import csrgraph
from csrgraph import methods
from nodevectors import ProNE
from tqdm import tqdm
import gc
import os
from time import perf_counter
import xxhash
import warnings
warnings.filterwarnings('ignore')

UINT32_MAX = (2**32) - 1
UINT16_MAX = (2**16) - 1


def get_prone_embeddings(
        G: csrgraph,
        prone_params: dict = {
            'n_components': 128
        },
        output_dir: str = 'embeddings'
    ):
    model = ProNE(**prone_params)

    embeddings = model.fit_transform(G)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    ## Write embeddings to npy file
    np.save(f'{output_dir}/embeddings.npy', embeddings)

    ## json dump name->idx mapping
    ## id_mapping = dict(zip(range(len(G.names)), G.names))
    id_mapping = dict(zip(G.names, range(len(G.names))))
    with open(f'{output_dir}/id_mapping.json', 'w') as f:
        json.dump(id_mapping, f)

    print(f'Saved embeddings and id_mapping to {output_dir}')

    return embeddings, id_mapping


def from_df(elist: pd.DataFrame, directed: bool = True) -> csrgraph:
    """
    Creates a csrgraph from a DataFrame of either two or three columns.

    elist :
        Either a DataFrame with two columns for source and target or three
        columns for source, target, and weight.
    Returns : csrgraph
    """
    if len(elist.columns) == 2:
        elist.columns = ['src', 'dst']
        elist['weight'] = np.ones(elist.shape[0])
    elif len(elist.columns) == 3:
        elist.columns = ['src', 'dst', 'weight']
    else: 
        raise ValueError(f"""
            Invalid columns: {elist.columns}
            Expected 2 (source, destination)
            or 3 (source, destination, weight)
            Read File: \n{elist.head(5)}
        """)
    # Create name mapping to normalize node IDs
    # Somehow this is 1.5x faster than np.union1d. Shame on numpy.
    allnodes = list(
        set(elist.src.unique())
        .union(set(elist.dst.unique())))
    # Factor all nodes to unique IDs
    names = (
        pd.Series(allnodes).astype('category')
        .cat.categories
    )
    nnodes = names.shape[0]

    # Get the input data type
    dtype = np.uint16
    if nnodes > UINT16_MAX:
        dtype = np.uint32
    if nnodes > UINT32_MAX:
        dtype = np.uint64

    name_dict = dict(zip(names,
                         np.arange(names.shape[0], dtype=dtype)))

    elist.src = elist.src.map(name_dict).astype(dtype)
    elist.dst = elist.dst.map(name_dict).astype(dtype)

    # clean up temp data
    allnodes = None
    name_dict = None
    gc.collect()

    # If undirected graph, append edgelist to reversed self
    if not directed:
        other_df = elist.copy()
        other_df.columns = ['dst', 'src', 'weight']
        elist = pd.concat([elist, other_df])
        other_df = None
        gc.collect()
    # Need to sort by src for _edgelist_to_graph
    elist = elist.sort_values(by='src')
    # extract numpy arrays and clear memory
    src = elist.src.to_numpy()
    dst = elist.dst.to_numpy()
    weight = elist.weight.to_numpy()
    elist = None
    gc.collect()
    G = methods._edgelist_to_graph(
        src, dst, weight,
        nnodes, nodenames=names
    )
    return G

def create_train_dataset(edgelist: pd.DataFrame, n_neg: int = 5):
    edgelist['src'] = edgelist['src'].astype(np.uint32)
    edgelist['dst'] = edgelist['dst'].astype(np.uint32)

    unique_src = edgelist['src'].unique()
    unique_dst = edgelist['dst'].unique()
    unique_node_ids = np.unique(np.concatenate([unique_src, unique_dst]))
    unique_node_ids_set = set(unique_node_ids)

    hash_func = lambda x: xxhash.xxh32(str(x)).intdigest()
    rand_ids = np.random.choice(unique_node_ids, size=len(unique_node_ids) // 10, replace=False)

    node_mapping = {}
    for value in rand_ids:
        hashed_value = hash_func(value)
        while hashed_value in unique_node_ids_set:
            hashed_value = hash_func(hashed_value)
        node_mapping[value] = hashed_value

    edgelist['src_train'] = edgelist['src'].map(node_mapping).fillna(edgelist['src']).astype(np.uint32)
    edgelist['dst_train'] = edgelist['dst'].map(node_mapping).fillna(edgelist['dst']).astype(np.uint32)

    pos_id_pairs = np.vstack([list(node_mapping.keys()), list(node_mapping.values())]).T

    unmapped_nodes = np.setdiff1d(unique_node_ids, list(node_mapping.keys()))
    neg_id_pairs = np.vstack([
        np.random.choice(unmapped_nodes, size=n_neg * len(pos_id_pairs)),
        np.random.choice(unmapped_nodes, size=n_neg * len(pos_id_pairs))
    ]).T

    return edgelist, pos_id_pairs, neg_id_pairs


if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    FILENAME = '../data/soc_livejournal_graph.feather'
    FILENAME = os.path.join(CURRENT_DIR, FILENAME)

    df = pd.read_feather(FILENAME)

    cols = ['from_node_id', 'to_node_id']

    ## np.random.seed(42)
    ## df = df.sample(10_000)

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
