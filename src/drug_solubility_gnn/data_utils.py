from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


ATOM_TYPES = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "Si",
    "B",
    "Se",
    "other",
]

HYBRIDIZATION_TYPES = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    "other",
]

BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]

MAX_DEGREE = 5


def _one_hot(value, categories: Sequence):
    return [1.0 if value == category else 0.0 for category in categories]


def _atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES:
        symbol = "other"
    atom_type_feature = _one_hot(symbol, ATOM_TYPES)

    degree = min(atom.GetDegree(), MAX_DEGREE)
    degree_feature = _one_hot(degree, list(range(MAX_DEGREE + 1)))

    formal_charge = float(atom.GetFormalCharge())

    hybridization = atom.GetHybridization()
    if hybridization not in HYBRIDIZATION_TYPES:
        hybridization = "other"
    hybridization_feature = _one_hot(hybridization, HYBRIDIZATION_TYPES)

    aromaticity = [1.0 if atom.GetIsAromatic() else 0.0]

    return atom_type_feature + degree_feature + [formal_charge] + hybridization_feature + aromaticity


def _bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    bond_type = bond.GetBondType()
    return _one_hot(bond_type, BOND_TYPES)


def smiles_to_data(smiles: str, target: float) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [_atom_features(atom) for atom in mol.GetAtoms()]
    if len(node_features) == 0:
        return None

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        edge_feature = _bond_features(bond)

        edge_indices.append([begin_idx, end_idx])
        edge_indices.append([end_idx, begin_idx])
        edge_attrs.append(edge_feature)
        edge_attrs.append(edge_feature)

    if len(edge_indices) == 0:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, len(BOND_TYPES)), dtype=torch.float)
    else:
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float)

    x_tensor = torch.tensor(node_features, dtype=torch.float)
    y_tensor = torch.tensor([target], dtype=torch.float)

    return Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y_tensor, smiles=smiles)


def infer_dataset_columns(df: pd.DataFrame) -> Tuple[str, str]:
    lower_map = {col.lower(): col for col in df.columns}

    smiles_col = lower_map.get("smiles")
    if smiles_col is None:
        raise ValueError("Cannot find SMILES column in dataset.")

    target_col = None
    for candidate in ["solubility", "logs", "logs", "log_s"]:
        if candidate in lower_map:
            target_col = lower_map[candidate]
            break

    if target_col is None:
        raise ValueError("Cannot find solubility/LogS target column in dataset.")

    return smiles_col, target_col


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)
    smiles_col, target_col = infer_dataset_columns(raw_df)
    dataset_df = raw_df[[smiles_col, target_col]].copy()
    dataset_df.columns = ["SMILES", "LogS"]
    dataset_df = dataset_df.dropna(subset=["SMILES", "LogS"]).reset_index(drop=True)
    dataset_df["LogS"] = dataset_df["LogS"].astype(float)
    return dataset_df


def build_graph_dataset(dataset_df: pd.DataFrame) -> Tuple[List[Data], List[int]]:
    graph_data_list: List[Data] = []
    valid_indices: List[int] = []

    for row_idx, row in dataset_df.iterrows():
        data = smiles_to_data(smiles=row["SMILES"], target=float(row["LogS"]))
        if data is None:
            continue
        graph_data_list.append(data)
        valid_indices.append(int(row_idx))

    if not graph_data_list:
        raise ValueError("No valid molecular graphs were generated from dataset.")

    return graph_data_list, valid_indices


def create_data_splits(
    graph_data_list: Sequence[Data],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, List[int]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    all_indices = list(range(len(graph_data_list)))

    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        shuffle=True,
    )

    val_relative = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1.0 - val_relative),
        random_state=seed,
        shuffle=True,
    )

    return {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
    }


def get_split_datasets(graph_data_list: Sequence[Data], split_indices: Dict[str, List[int]]):
    train_dataset = [graph_data_list[i] for i in split_indices["train"]]
    val_dataset = [graph_data_list[i] for i in split_indices["val"]]
    test_dataset = [graph_data_list[i] for i in split_indices["test"]]
    return train_dataset, val_dataset, test_dataset
