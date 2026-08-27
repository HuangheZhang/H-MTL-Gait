"""
Gait Analysis Dataset
=====================
Unified data loading and preprocessing for multi-task gait analysis.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from .normalization import FoldStandardizer


class GaitDataset(Dataset):
    """PyTorch Dataset for gait analysis."""
    
    def __init__(
        self,
        data: np.ndarray,
        labels: Dict[str, np.ndarray],
        metadata: Optional[pd.DataFrame] = None,
        transform=None
    ):
        """
        Args:
            data: Sensor data of shape (N, T, C) or (N, C, T)
            labels: Dictionary of task labels
            metadata: Optional metadata DataFrame
            transform: Optional transforms
        """
        self.data = torch.FloatTensor(data)
        # Regression tasks use FloatTensor, classification tasks use LongTensor
        REGRESSION_TASKS = {'regression', 'vga_regression', 'age', 'tug'}
        self.labels = {k: torch.FloatTensor(v) if k in REGRESSION_TASKS else torch.LongTensor(v) 
                       for k, v in labels.items()}
        self.metadata = metadata
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        
        if self.transform:
            x = self.transform(x)
            
        labels = {k: v[idx] for k, v in self.labels.items()}
        
        return x, labels


class GaitDataLoader:
    """Main data loader for the gait analysis dataset."""
    
    # Class mappings
    PATHOLOGY_TO_GROUP = {
        'HS': 'healthy',
        'CVA': 'neuro', 'PD': 'neuro', 'CIPN': 'neuro', 'RIL': 'neuro',
        'KOA': 'ortho', 'HOA': 'ortho', 'ACL': 'ortho'
    }
    
    GROUP_TO_ID = {'healthy': 0, 'neuro': 1, 'ortho': 2}
    
    PATHOLOGY_TO_ID = {
        'HS': 0, 'CVA': 1, 'PD': 2, 'CIPN': 3, 
        'RIL': 4, 'KOA': 5, 'HOA': 6, 'ACL': 7
    }
    
    CLINICAL_SCORES = {
        'CVA': ('FMA-LE', 34),
        'PD': ('UPDRS III', 108),
        'CIPN': ('TNSc', 28),
        'RIL': ('mSS', 30),
        'KOA': ('WOMAC', 100),
        'HOA': ('WOMAC', 100),
        'ACL': ('IKDC', 100)
    }
    
    GENDER_TO_ID = {'M': 0, 'F': 1}
    
    NEURO_PATHOLOGY_TO_ID = {'CVA': 0, 'PD': 1, 'CIPN': 2, 'RIL': 3}
    
    MAX_AGE = 100.0  # for normalization
    MAX_TUG = 100.0  # for normalization (max observed ~90s)
    MAX_VGA = 4.0    # visual gait assessment max score
    
    SENSORS = ['HE', 'LB', 'LF', 'RF']
    
    SIGNALS = {
        'acc': ['Acc_X', 'Acc_Y', 'Acc_Z'],
        'freeacc': ['FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z'],
        'gyr': ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
    }
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: Path to the data directory
        """
        self.base_path = Path(base_path)
        self.data_cache = {}
        
    def load_trial(self, trial_path: Path) -> Dict:
        """Load a single trial's data and metadata."""
        trial_name = trial_path.name
        
        # Load processed data
        processed_file = trial_path / f"{trial_name}_processed_data.txt"
        if not processed_file.exists():
            return None
            
        data = pd.read_csv(processed_file, sep='\t')
        
        # Load metadata
        meta_file = trial_path / f"{trial_name}_meta.json"
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
            
        return {
            'data': data,
            'metadata': metadata,
            'trial_name': trial_name
        }
    
    def load_all_data(self, verbose: bool = True) -> Tuple[List, List]:
        """Load all trials from all cohorts."""
        all_trials = []
        all_metadata = []
        
        # Iterate through cohorts
        for group in ['healthy', 'neuro', 'ortho']:
            group_path = self.base_path / group
            if not group_path.exists():
                continue
                
            for pathology_dir in group_path.iterdir():
                if not pathology_dir.is_dir():
                    continue
                    
                pathology = pathology_dir.name
                
                for subject_dir in pathology_dir.iterdir():
                    if not subject_dir.is_dir():
                        continue
                        
                    for trial_dir in subject_dir.iterdir():
                        if not trial_dir.is_dir():
                            continue
                            
                        trial = self.load_trial(trial_dir)
                        if trial is not None:
                            all_trials.append(trial)
                            all_metadata.append({
                                'group': group,
                                'pathology': pathology,
                                'subject': trial['metadata']['subject'],
                                'trial': trial['trial_name'],
                                **{k: v for k, v in trial['metadata'].items() 
                                   if k in ['age', 'gender', 'height', 'weight', 'BMI',
                                           'evaluationScoreValue', 'evaluationScoreName',
                                           'visualGaitAssessment', 'TUG']}
                            })
        
        if verbose:
            print(f"Loaded {len(all_trials)} trials")
            
        return all_trials, pd.DataFrame(all_metadata)
    
    def get_column_names(
        self, 
        sensors: List[str] = None,
        signals: List[str] = None
    ) -> List[str]:
        """Get column names for specified sensors and signals."""
        sensors = sensors or self.SENSORS
        signals = signals or list(self.SIGNALS.keys())
        
        columns = []
        for sensor in sensors:
            for signal_type in signals:
                for signal_name in self.SIGNALS[signal_type]:
                    columns.append(f"{sensor}_{signal_name}")
                    
        return columns
    
    def extract_windows(
        self,
        data: pd.DataFrame,
        window_size: int = 200,
        stride: int = 100,
        columns: List[str] = None
    ) -> np.ndarray:
        """Extract sliding windows from time series data."""
        if columns:
            data = data[columns]
        else:
            # Remove PacketCounter
            data = data.drop(columns=['PacketCounter'], errors='ignore')
            
        values = data.values
        n_samples = len(values)
        
        windows = []
        for start in range(0, n_samples - window_size + 1, stride):
            window = values[start:start + window_size]
            windows.append(window)
            
        return np.array(windows)
    
    def prepare_dataset(
        self,
        sensors: List[str] = None,
        signals: List[str] = None,
        window_size: int = 200,
        stride: int = 100,
        normalize: bool = False,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
        """
        Prepare the full dataset for training.
        
        Returns:
            X: Data array of shape (N, T, C)
            y: Dictionary of labels for each task
            metadata: DataFrame with sample metadata
        """
        if normalize:
            raise ValueError(
                "Normalization must occur after subject splitting via "
                "FoldStandardizer; prepare_dataset only returns raw windows."
            )

        trials, meta_df = self.load_all_data(verbose=verbose)
        
        # Get columns to use
        columns = self.get_column_names(sensors, signals)
        
        all_windows = []
        all_labels = defaultdict(list)
        all_meta = []
        
        for trial, meta in zip(trials, meta_df.to_dict('records')):
            # Check if all columns exist
            available_cols = [c for c in columns if c in trial['data'].columns]
            if len(available_cols) != len(columns):
                warnings.warn(f"Missing columns in {trial['trial_name']}")
                continue
                
            # Extract windows
            windows = self.extract_windows(
                trial['data'], 
                window_size=window_size,
                stride=stride,
                columns=columns
            )
            
            if len(windows) == 0:
                continue
                
            all_windows.append(windows)
            
            # Prepare labels for each window
            pathology = meta['pathology']
            n_windows = len(windows)
            
            # Task 1: Binary (healthy vs pathological)
            binary_label = 0 if pathology == 'HS' else 1
            all_labels['binary'].extend([binary_label] * n_windows)
            
            # Task 2: Coarse (3-class)
            coarse_label = self.GROUP_TO_ID[self.PATHOLOGY_TO_GROUP[pathology]]
            all_labels['coarse'].extend([coarse_label] * n_windows)
            
            # Task 3: Fine (8-class)
            fine_label = self.PATHOLOGY_TO_ID[pathology]
            all_labels['fine'].extend([fine_label] * n_windows)
            
            # Task 4: Regression (clinical score)
            score = meta.get('evaluationScoreValue')
            if score is not None and pathology in self.CLINICAL_SCORES:
                max_score = self.CLINICAL_SCORES[pathology][1]
                try:
                    normalized_score = float(score) / max_score  # Normalize to [0, 1]
                except (ValueError, TypeError):
                    normalized_score = -1  # Mark as missing (e.g., "Not evaluated")
            else:
                normalized_score = -1  # Mark as missing
            all_labels['regression'].extend([normalized_score] * n_windows)
            
            # Task 5: VGA classification (5-class: 0-4)
            vga = meta.get('visualGaitAssessment')
            if vga is not None:
                try:
                    vga_label = int(float(vga))
                    vga_label = max(0, min(4, vga_label))  # clamp to [0,4]
                except (ValueError, TypeError):
                    vga_label = -1
            else:
                vga_label = -1
            all_labels['vga_class'].extend([vga_label] * n_windows)
            
            # Task 6: VGA regression (normalized /4)
            if vga is not None and vga_label >= 0:
                vga_reg = float(vga) / self.MAX_VGA
            else:
                vga_reg = -1
            all_labels['vga_regression'].extend([vga_reg] * n_windows)
            
            # Task 7: Gender classification (M=0, F=1)
            gender = meta.get('gender')
            gender_label = self.GENDER_TO_ID.get(gender, -1)
            all_labels['gender'].extend([gender_label] * n_windows)
            
            # Task 8: Age regression (normalized /100)
            age = meta.get('age')
            if age is not None and not (isinstance(age, float) and np.isnan(age)):
                try:
                    age_reg = float(age) / self.MAX_AGE
                except (ValueError, TypeError):
                    age_reg = -1
            else:
                age_reg = -1
            all_labels['age'].extend([age_reg] * n_windows)
            
            # Task 9: TUG regression (normalized /100)
            tug = meta.get('TUG')
            if tug is not None and tug != 'Not evaluated':
                try:
                    tug_reg = float(tug) / self.MAX_TUG
                except (ValueError, TypeError):
                    tug_reg = -1
            else:
                tug_reg = -1
            all_labels['tug'].extend([tug_reg] * n_windows)
            
            # Task 10: Neuro fine classification (4-class, only for neuro patients)
            if pathology in self.NEURO_PATHOLOGY_TO_ID:
                neuro_fine_label = self.NEURO_PATHOLOGY_TO_ID[pathology]
            else:
                neuro_fine_label = -1  # Mark non-neuro as missing
            all_labels['neuro_fine'].extend([neuro_fine_label] * n_windows)
            
            # Store metadata for each window
            for _ in range(n_windows):
                all_meta.append(meta)
        
        # Combine all windows
        X = np.concatenate(all_windows, axis=0)  # (N, T, C)
        
        # Convert labels
        y = {k: np.array(v) for k, v in all_labels.items()}
        
        # Metadata DataFrame
        metadata = pd.DataFrame(all_meta)
        
        if verbose:
            print(f"Dataset shape: {X.shape}")
            print(f"Labels: {[(k, v.shape) for k, v in y.items()]}")
            
        return X, y, metadata
    
    def create_dataloaders(
        self,
        X: np.ndarray,
        y: Dict[str, np.ndarray],
        metadata: pd.DataFrame,
        batch_size: int = 32,
        test_size: float = 0.2,
        val_size: float = 0.1,
        subject_wise: bool = True,
        random_state: int = 42,
        normalize: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create deterministic stratified loaders with training-only scaling."""
        inner_val_size = val_size / (1.0 - test_size)

        if subject_wise:
            subjects = metadata['subject'].unique()
            subject_labels = np.array([
                metadata.loc[metadata['subject'] == subject, 'pathology'].iloc[0]
                for subject in subjects
            ])
            outer_split = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_val_pos, test_pos = next(
                outer_split.split(subjects, subject_labels)
            )
            train_val_subjects = subjects[train_val_pos]
            inner_split = StratifiedShuffleSplit(
                n_splits=1, test_size=inner_val_size,
                random_state=random_state + 1
            )
            train_pos, val_pos = next(inner_split.split(
                train_val_subjects, subject_labels[train_val_pos]
            ))
            train_subjects = train_val_subjects[train_pos]
            val_subjects = train_val_subjects[val_pos]
            test_subjects = subjects[test_pos]
            train_idx = np.flatnonzero(
                metadata['subject'].isin(train_subjects).to_numpy()
            )
            val_idx = np.flatnonzero(
                metadata['subject'].isin(val_subjects).to_numpy()
            )
            test_idx = np.flatnonzero(
                metadata['subject'].isin(test_subjects).to_numpy()
            )
        else:
            indices = np.arange(len(X))
            outer_split = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_val_idx, test_idx = next(
                outer_split.split(indices, y['fine'])
            )
            inner_split = StratifiedShuffleSplit(
                n_splits=1, test_size=inner_val_size,
                random_state=random_state + 1
            )
            train_pos, val_pos = next(
                inner_split.split(train_val_idx, y['fine'][train_val_idx])
            )
            train_idx = train_val_idx[train_pos]
            val_idx = train_val_idx[val_pos]

        train_data = X[train_idx]
        val_data = X[val_idx]
        test_data = X[test_idx]
        if normalize:
            standardizer = FoldStandardizer()
            train_data = standardizer.fit_transform(train_data)
            val_data = standardizer.transform(val_data)
            test_data = standardizer.transform(test_data)

        def make_dataset(indices, data):
            return GaitDataset(
                data,
                {name: values[indices] for name, values in y.items()},
                metadata.iloc[indices].reset_index(drop=True)
            )

        train_dataset = make_dataset(train_idx, train_data)
        val_dataset = make_dataset(val_idx, val_data)
        test_dataset = make_dataset(test_idx, test_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


def get_dataset_statistics(base_path: str) -> pd.DataFrame:
    """Get comprehensive dataset statistics."""
    loader = GaitDataLoader(base_path)
    _, metadata = loader.load_all_data(verbose=False)
    
    stats = []
    for pathology in loader.PATHOLOGY_TO_ID.keys():
        subset = metadata[metadata['pathology'] == pathology]
        
        stats.append({
            'Pathology': pathology,
            'Group': loader.PATHOLOGY_TO_GROUP[pathology],
            'Subjects': subset['subject'].nunique(),
            'Trials': len(subset),
            'Age (mean+/-std)': f"{subset['age'].mean():.1f}+/-{subset['age'].std():.1f}",
            'Gender (M/F)': f"{(subset['gender']=='M').sum()}/{(subset['gender']=='F').sum()}",
            'BMI (mean+/-std)': f"{subset['BMI'].mean():.1f}+/-{subset['BMI'].std():.1f}"
        })
        
    return pd.DataFrame(stats)


if __name__ == "__main__":
    # Example usage
    base_path = "data"
    
    # Get statistics
    stats = get_dataset_statistics(base_path)
    print(stats.to_string())
    
    # Load and prepare dataset
    loader = GaitDataLoader(base_path)
    X, y, metadata = loader.prepare_dataset(
        sensors=['LB', 'LF', 'RF'],  # Example: without head sensor
        signals=['freeacc', 'gyr'],   # Example: optimal signal combination
        window_size=200,
        stride=100
    )
    
    print(f"\nPrepared dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  Binary labels distribution: {np.bincount(y['binary'])}")
    print(f"  Coarse labels distribution: {np.bincount(y['coarse'])}")
    print(f"  Fine labels distribution: {np.bincount(y['fine'])}")
