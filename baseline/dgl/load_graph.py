import os
import pickle

import numpy as np
import torch
from ogb.lsc import MAG240MDataset
from ogb.nodeproppred import DglNodePropPredDataset

import dgl
import dgl.function as fn
from dgl.data import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
from IGBDataset import IGBHeteroDGLDataset
from ohgb_dataset import OHGBDataset
import gc, time


# def _com_pca_sim(before, after, path):
#     u,s,v=after
#     # rank=torch.linalg.matrix_rank(before)
#     # print(f'feat.rank={rank}')
#     print(f"u: {u.shape}, s: {torch.diag(s).shape}, v: {v.H.shape}")
#     tmp=torch.matmul(u, torch.diag(s))
#     tmp=torch.matmul(tmp, v.H)
#     # xx=torch.nn.MSELoss()
#     # loss=xx(before, tmp)
#     # print('loss=',loss)
#     print(f"A'.shape: {tmp.shape}")
#     print(f"A.shape: {before.shape}")
#     assert tmp.shape[0]==before.shape[0] and tmp.shape[1]==before.shape[1], 'dim_error!'
#     sim_dict=_cmp_cos_sim({'sim':before}, {'sim':tmp})
#     cdf_tensor_dict, total_tensor=_cul_cdf(sim_dict)
#     draw_cdf(cdf_tensor_dict, path)

def load_ogbn_mag(root: str = 'dataset', complete_missing_feats: bool = False):
    """
    Load the ogbn-mag dataset, and return the graph, train_idx, val_idx, test_idx, and
    num_classes and predict_category.
    """
    dataset = DglNodePropPredDataset("ogbn-mag", root=root)
    graph, labels = dataset[0]
    labels["paper"] = labels["paper"][:, 0]
    graph.ndata["label"] = labels
    # add reverse edges in "cites" relation, and add reverse edge types for the rest etypes
    graph = dgl.AddReverse()(graph)

    if complete_missing_feats:
        # precompute the author, topic, and institution features
        graph.update_all(
            fn.copy_u("feat", "m"), fn.mean("m", "feat"), etype="rev_writes"
        )
        graph.update_all(
            fn.copy_u("feat", "m"), fn.mean("m", "feat"), etype="has_topic"
        )
        graph.update_all(
            fn.copy_u("feat", "m"), fn.mean("m", "feat"), etype="affiliated_with"
        )

    # find train/val/test indexes
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )

    predict_category = "paper"

    graph.ndata["train_mask"] = {predict_category: torch.zeros(
        graph.number_of_nodes(predict_category), dtype=torch.bool)}
    graph.ndata["val_mask"] = {predict_category: torch.zeros(
        graph.number_of_nodes(predict_category), dtype=torch.bool)}
    graph.ndata["test_mask"] = {predict_category: torch.zeros(
        graph.number_of_nodes(predict_category), dtype=torch.bool)}

    graph.ndata["train_mask"][predict_category][train_idx[predict_category]] = True
    graph.ndata['val_mask'][predict_category][val_idx[predict_category]] = True
    graph.ndata['test_mask'][predict_category][test_idx[predict_category]] = True

    return graph, dataset.num_classes, predict_category, None, "rev_"


def load_dgl_data(dataset_name: str, complete_missing_feats: bool = False):
    """
    Load the aifb dataset, and return the graph, train_idx, val_idx, test_idx, and
    num_classes and predict_category.
    """
    if dataset_name == "aifb":
        dataset = AIFBDataset()
    elif dataset_name == "am":
        dataset = AMDataset()
    elif dataset_name == "bgs":
        dataset = BGSDataset()
    elif dataset_name == "mutag":
        dataset = MUTAGDataset()
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))

    graph = dataset[0]
    graph = dgl.AddReverse()(graph)
    predict_category = dataset.predict_category

    if complete_missing_feats:
        # this is featureless graph, give random feature for each ntype
        graph.ndata["feat"] = {ntype: torch.randn(
            graph.number_of_nodes(ntype), 32) for ntype in graph.ntypes}

    train_mask = graph.nodes[predict_category].data.pop("train_mask")
    test_mask = graph.nodes[predict_category].data.pop("test_mask")

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    val_idx = train_idx[: len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5:]

    train_idx = {predict_category: train_idx}
    val_idx = {predict_category: val_idx}
    test_idx = {predict_category: test_idx}

    graph.ndata["train_mask"] = {predict_category: torch.zeros(
        graph.number_of_nodes(predict_category), dtype=torch.bool)}
    graph.ndata["val_mask"] = {predict_category: torch.zeros(
        graph.number_of_nodes(predict_category), dtype=torch.bool)}
    graph.ndata["test_mask"] = {predict_category: torch.zeros(
        graph.number_of_nodes(predict_category), dtype=torch.bool)}

    graph.ndata["train_mask"][predict_category][train_idx[predict_category]] = True
    graph.ndata['val_mask'][predict_category][val_idx[predict_category]] = True
    graph.ndata['test_mask'][predict_category][test_idx[predict_category]] = True

    return graph, dataset.num_classes, predict_category, None, "rev_"


def load_mag240m(root: str = 'dataset', load_feat: bool = True):
    dataset = MAG240MDataset(root=root)
    ei_writes = dataset.edge_index("author", "writes", "paper")
    ei_cites = dataset.edge_index("paper", "paper")
    ei_affiliated = dataset.edge_index("author", "institution")

    g = dgl.heterograph(
        {
            ("author", "writes", "paper"): (ei_writes[0], ei_writes[1]),
            ("paper", "rev_writes", "author"): (ei_writes[1], ei_writes[0]),
            ("author", "affiliated_with", "institution"): (
                ei_affiliated[0],
                ei_affiliated[1],
            ),
            ("institution", "rev_affiliated_with", "author"): (
                ei_affiliated[1],
                ei_affiliated[0],
            ),
            ("paper", "cites", "paper"): (
                np.concatenate([ei_cites[0], ei_cites[1]]),
                np.concatenate([ei_cites[1], ei_cites[0]]),
            ),
        }
    )
    g.ndata['label'] = {'paper': torch.from_numpy(dataset.all_paper_label)}
    if load_feat:
        # g.ndata['feat'] = {'paper': torch.from_numpy(dataset.all_paper_feat)}
        g.ndata['feat'] = {'paper': torch.from_numpy(dataset.paper_feat)}

    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test-dev"],
    )

    predict_category = "paper"

    g.ndata["train_mask"] = {predict_category: torch.zeros(
        g.number_of_nodes(predict_category), dtype=torch.bool)}
    g.ndata["val_mask"] = {predict_category: torch.zeros(
        g.number_of_nodes(predict_category), dtype=torch.bool)}
    g.ndata["test_mask"] = {predict_category: torch.zeros(
        g.number_of_nodes(predict_category), dtype=torch.bool)}

    g.ndata["train_mask"][predict_category][train_idx] = True
    g.ndata['val_mask'][predict_category][val_idx] = True
    g.ndata['test_mask'][predict_category][test_idx] = True

    return g, dataset.num_classes, predict_category, None, "rev_"


def load_freebase(root: str = 'dataset'):
    dataset = OHGBDataset("ohgbn-Freebase", raw_dir=root)
    g = dataset[0]
    predict_category = "BOOK"
    g = dgl.AddReverse()(g)
    num_classes = 8
    list_of_metapaths = None
    return g, num_classes, predict_category, list_of_metapaths, "rev_"


def load_igb_het(root: str = 'dataset', dataset_size: str='small', load_feat: bool = True):
    dataset = IGBHeteroDGLDataset(dir=root, dataset_size=dataset_size,
                                  num_classes=2983, in_memory=load_feat, all_in_edges=True)
    g = dataset[0]
    g = dgl.AddReverse()(g)
    predict_category = "paper"
    num_classes = 2983
    list_of_metapaths = None
    return g, num_classes, predict_category, list_of_metapaths, "rev_"


def load_igb_full(dataset:str='small', root:str='dataset', use19: bool =True, load_feat: bool = True):
    paper_paper_edges = torch.from_numpy(np.load(os.path.join(root, dataset, 'processed',
    'paper__cites__paper', 'edge_index.npy')))
    author_paper_edges = torch.from_numpy(np.load(os.path.join(root, dataset, 'processed', 
    'paper__written_by__author', 'edge_index.npy')))
    affiliation_author_edges = torch.from_numpy(np.load(os.path.join(root, dataset, 'processed', 
    'author__affiliated_to__institute', 'edge_index.npy')))
    paper_fos_edges = torch.from_numpy(np.load(os.path.join(root, dataset, 'processed', 
    'paper__topic__fos', 'edge_index.npy')))
    paper_journal_edges = torch.from_numpy(np.load(os.path.join(root, dataset, 'processed', 
    'paper__published__journal', 'edge_index.npy')))
    paper_conference_edges = torch.from_numpy(np.load(os.path.join(root, dataset, 'processed', 
    'paper__venue__conference', 'edge_index.npy')))

    graph_data = {
        ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
        ('paper', 'rev_writes', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
        ('author', 'writes', 'paper'): (author_paper_edges[:, 1], author_paper_edges[:, 0]),
        ('author', 'affiliated_with', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
        ('institute', 'rev_affiliated_with', 'author'): (affiliation_author_edges[:, 1], affiliation_author_edges[:, 0]),
        ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1]),
        ('fos', 'rev_topic', 'paper'): (paper_fos_edges[:, 1], paper_fos_edges[:, 0]),
        ('paper', 'published', 'journal'): (paper_journal_edges[:, 0], paper_journal_edges[:, 1]),
        ('journal', 'rev_published', 'paper'): (paper_journal_edges[:, 1], paper_journal_edges[:, 0]),
        ('paper', 'venue', 'conference'): (paper_conference_edges[:, 0], paper_conference_edges[:, 1]),
        ('conference', 'rev_venue', 'paper'): (paper_conference_edges[:, 1], paper_conference_edges[:, 0]),
    }

    hg = dgl.heterograph(graph_data)     

    label_name='node_label_19.npy' if use19 else 'node_label_2K.npy'
    paper_node_labels = torch.from_numpy(np.load(os.path.join(root, dataset, 'processed', 'paper', label_name)))
    print("label:", paper_node_labels.squeeze())
    hg.nodes['paper'].data['label'] = paper_node_labels.squeeze()

    def _load_feat(name):
        feat = np.load(os.path.join(root, dataset, 'processed', name, 'node_feat.npy'))
        print(f'feat1 shape:', feat.shape)
        hg.nodes[name].data['feat'] = torch.from_numpy(feat)
        return feat.shape[0]

    if load_feat:
        hg.num_paper_nodes=_load_feat('paper')
        hg.num_author_nodes=_load_feat('author')
        hg.num_institute_nodes=_load_feat('institute')
        hg.num_fos_nodes=_load_feat('fos')
        hg.num_conference_nodes=_load_feat('conference')
        hg.num_journal_nodes=_load_feat('journal')

    hg = dgl.remove_self_loop(hg, etype='cites')
    hg = dgl.add_self_loop(hg, etype='cites')

    n_nodes = 1000000 if dataset=='hete_small' else 10000000

    if dataset=="medium":
        n_train = int(n_nodes * 0.1)
        n_val = int(n_nodes * 0.4)
    else:
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    
    hg.nodes['paper'].data['train_mask'] = train_mask
    hg.nodes['paper'].data['val_mask'] = val_mask
    hg.nodes['paper'].data['test_mask'] = test_mask

    num_rels = len(hg.canonical_etypes)
    num_of_ntype = len(hg.ntypes)
    num_classes = 19 if use19 else 2983

    print('Number of relations: {}'.format(num_rels))
    print('Number of class: {}'.format(num_classes))
    print('Number of train: {}'.format(n_train))
    print('Number of valid: {}'.format(n_val))
    print('Number of test: {}'.format(n_nodes*0.2))

    # get target category id
    category = 'paper'
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    list_of_metapaths=None

    return hg, num_classes, category, list_of_metapaths, "rev_"

def load_donor(root: str = 'dataset', load_feat: bool = True):
    g = dgl.load_graphs(os.patorch.join(root, 'donor2', 'graph.bin'))[0][0]
    predict_category = "Project"
    num_classes = 2
    list_of_metapaths = None

    if load_feat:
        node_feats = torch.load(os.patorch.join(root, 'donor2', 'node_feats.pt'))
        g.ndata['feat'] = {
            ntype: node_feats[ntype]['feat'] for ntype in g.ntypes
        }

    train = torch.load(os.patorch.join(root, 'donor2', 'train.pt'))
    val = torch.load(os.patorch.join(root, 'donor2', 'valid.pt'))
    test = torch.load(os.patorch.join(root, 'donor2', 'test.pt'))

    train_idx, train_label = train[predict_category]['id'], train[predict_category]['label']
    val_idx, val_label = val[predict_category]['id'], val[predict_category]['label']
    test_idx, test_label = test[predict_category]['id'], test[predict_category]['label']

    labels = torch.zeros(g.number_of_nodes(predict_category), dtype=torch.long)
    labels[train_idx] = train_label
    labels[val_idx] = val_label
    labels[test_idx] = test_label

    g.ndata['label'] = {predict_category: labels}

    g.ndata["train_mask"] = {predict_category: torch.zeros(
        g.number_of_nodes(predict_category), dtype=torch.bool)}
    g.ndata["val_mask"] = {predict_category: torch.zeros(
        g.number_of_nodes(predict_category), dtype=torch.bool)}
    g.ndata["test_mask"] = {predict_category: torch.zeros(
        g.number_of_nodes(predict_category), dtype=torch.bool)}

    g.ndata["train_mask"][predict_category][train_idx] = True
    g.ndata['val_mask'][predict_category][val_idx] = True
    g.ndata['test_mask'][predict_category][test_idx] = True

    return g, num_classes, predict_category, list_of_metapaths, "reverse_"


def load_stackexchange(root: str = 'dataset', load_feat: bool = True):
    # 定义边文件的路径和对应的关系
    edges_files = {
        ("Badges", "Badges-UserId", "Users"): "Badges:Badges-UserId:Users_edges.npy",
        ("Comments", "Comments-PostId", "Posts"): "Comments:Comments-PostId:Posts_edges.npy",
        ("Comments", "Comments-UserId", "Users"): "Comments:Comments-UserId:Users_edges.npy",
        ("PostHistory", "PostHistory-PostId", "Posts"): "PostHistory:PostHistory-PostId:Posts_edges.npy",
        ("PostHistory", "PostHistory-UserId", "Users"): "PostHistory:PostHistory-UserId:Users_edges.npy",
        ("PostLink", "PostLink-PostId", "Posts"): "PostLink:PostLink-PostId:Posts_edges.npy",
        ("PostLink", "PostLink-RelatedPostId", "Posts"): "PostLink:PostLink-RelatedPostId:Posts_edges.npy",
        ("Posts", "Posts-AcceptedAnswerId", "Posts"): "Posts:Posts-AcceptedAnswerId:Posts_edges.npy",
        ("Posts", "Posts-LastEditorUserId", "Users"): "Posts:Posts-LastEditorUserId:Users_edges.npy",
        ("Posts", "Posts-OwnerUserId", "Users"): "Posts:Posts-OwnerUserId:Users_edges.npy",
        ("Posts", "Posts-ParentId", "Posts"): "Posts:Posts-ParentId:Posts_edges.npy",
        ("Posts", "reverse_Comments-PostId", "Comments"): "Posts:reverse_Comments-PostId:Comments_edges.npy",
        ("Posts", "reverse_PostHistory-PostId", "PostHistory"): "Posts:reverse_PostHistory-PostId:PostHistory_edges.npy",
        ("Posts", "reverse_PostLink-PostId", "PostLink"): "Posts:reverse_PostLink-PostId:PostLink_edges.npy",
        ("Posts", "reverse_PostLink-RelatedPostId", "PostLink"): "Posts:reverse_PostLink-RelatedPostId:PostLink_edges.npy",
        ("Posts", "reverse_PostTag-PostId", "PostTag"): "Posts:reverse_PostTag-PostId:PostTag_edges.npy",
        ("Posts", "reverse_Tag-ExcerptPostId", "Tag"): "Posts:reverse_Tag-ExcerptPostId:Tag_edges.npy",
        ("Posts", "reverse_Tag-WikiPostId", "Tag"): "Posts:reverse_Tag-WikiPostId:Tag_edges.npy",
        ("Posts", "reverse_Vote-PostId", "Vote"): "Posts:reverse_Vote-PostId:Vote_edges.npy",
        ("PostTag", "PostTag-PostId", "Posts"): "PostTag:PostTag-PostId:Posts_edges.npy",
        ("PostTag", "PostTag-TagId", "Tag"): "PostTag:PostTag-TagId:Tag_edges.npy",
        ("Tag", "reverse_PostTag-TagId", "PostTag"): "Tag:reverse_PostTag-TagId:PostTag_edges.npy",
        ("Tag", "Tag-ExcerptPostId", "Posts"): "Tag:Tag-ExcerptPostId:Posts_edges.npy",
        ("Tag", "Tag-WikiPostId", "Posts"): "Tag:Tag-WikiPostId:Posts_edges.npy",
        ("Users", "reverse_Badges-UserId", "Badges"): "Users:reverse_Badges-UserId:Badges_edges.npy",
        ("Users", "reverse_Comments-UserId", "Comments"): "Users:reverse_Comments-UserId:Comments_edges.npy",
        ("Users", "reverse_PostHistory-UserId", "PostHistory"): "Users:reverse_PostHistory-UserId:PostHistory_edges.npy",
        ("Users", "reverse_Posts-LastEditorUserId", "Posts"): "Users:reverse_Posts-LastEditorUserId:Posts_edges.npy",
        ("Users", "reverse_Posts-OwnerUserId", "Posts"): "Users:reverse_Posts-OwnerUserId:Posts_edges.npy",
        ("Users", "reverse_Vote-UserId", "Vote"): "Users:reverse_Vote-UserId:Vote_edges.npy",
        ("Vote", "Vote-PostId", "Posts"): "Vote:Vote-PostId:Posts_edges.npy",
        ("Vote", "Vote-UserId", "Users"): "Vote:Vote-UserId:Users_edges.npy"
    }

    predict_category = "Users"
    num_classes = 2
    list_of_metapaths = None

    # 创建一个包含所有边数据的字典
    graph_data = {}
    for (src_type, etype, dst_type), file_path in edges_files.items():
        edges = np.load(os.patorch.join(
            root, "stackexchange-r2n", "edges", file_path))
        src, dst = edges[0], edges[1]
        graph_data[(src_type, etype, dst_type)] = (src, dst)

    # 创建异构图
    g = dgl.heterograph(graph_data)

    # 加载节点特征
    if load_feat:
        for ntype in g.ntypes:
            feat = torch.from_numpy(np.load(os.patorch.join(
                root, "stackexchange-r2n", "features", f"{ntype}.npy")))
            g.nodes[ntype].data['feat'] = feat

    # 加载标签
    train_nodes = np.load(os.patorch.join(
        root, "stackexchange-r2n", "churn", "train_set", "Users_seed_nodes.npy"))
    val_nodes = np.load(os.patorch.join(root, "stackexchange-r2n",
                        "churn", "validation_set", "Users_seed_nodes.npy"))
    test_nodes = np.load(os.patorch.join(
        root, "stackexchange-r2n", "churn", "test_set", "Users_seed_nodes.npy"))

    g.ndata['train_mask'] = {predict_category: torch.zeros(
        g.number_of_nodes(predict_category), dtype=torch.bool)}
    g.ndata['val_mask'] = {predict_category: torch.zeros(
        g.number_of_nodes(predict_category), dtype=torch.bool)}
    g.ndata['test_mask'] = {predict_category: torch.zeros(
        g.number_of_nodes(predict_category), dtype=torch.bool)}

    g.ndata['train_mask'][predict_category][train_nodes] = True
    g.ndata['val_mask'][predict_category][val_nodes] = True
    g.ndata['test_mask'][predict_category][test_nodes] = True

    train_labels = np.load(os.patorch.join(
        root, "stackexchange-r2n", "churn", "train_set", "Users_labels.npy"))
    val_labels = np.load(os.patorch.join(
        root, "stackexchange-r2n", "churn", "validation_set", "Users_labels.npy"))
    test_labels = np.load(os.patorch.join(
        root, "stackexchange-r2n", "churn", "test_set", "Users_labels.npy"))

    labels = torch.zeros(g.number_of_nodes(predict_category), dtype=torch.long)
    labels[train_nodes] = torch.from_numpy(train_labels)
    labels[val_nodes] = torch.from_numpy(val_labels)
    labels[test_nodes] = torch.from_numpy(test_labels)

    g.ndata['label'] = {predict_category: labels}

    return g, num_classes, predict_category, list_of_metapaths, "reverse_"


def load_dataset(dataset: str, root: str = 'dataset', load_feat: bool = True, complete_missing_feats: bool = False):
    if dataset == "ogbn-mag":
        return load_ogbn_mag(root, complete_missing_feats)
    elif dataset == 'mag240m':# dont load feat
        return load_mag240m(root, False)
    elif dataset == "freebase":
        return load_freebase(root)
    elif dataset == 'igb-het':
        return load_igb_het(root, 'small',load_feat)
    elif dataset == 'donor':
        return load_donor(root, load_feat)
    elif dataset == 'stackexchange':
        return load_stackexchange(root, load_feat)
    elif dataset == 'igb-full-small':
        return load_igb_full('hete_small', root, use19=True, load_feat=load_feat)
    elif dataset == 'igb-full-medium':
        return load_igb_full('hete_medium', root, use19=True, load_feat=load_feat)
    else:
        return load_dgl_data(dataset, complete_missing_feats)

if __name__ == '__main__':
    g, num_classes, predict_category, list_of_metapaths, reverse_edge_type_prefix = load_dataset(
        'donor')
    import pdb
    pdb.set_trace()
