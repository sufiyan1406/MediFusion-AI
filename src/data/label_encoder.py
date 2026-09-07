"""
Label Encoder Module for MediFusion.

Parses NIH ChestX-ray14 pipe-separated disease string labels into 14-element multi-hot binary vectors.
Strictly validates all labels against the 14 expected pathology findings and ensures 'No Finding'
is represented as an all-zero target vector.
"""

from typing import List, Union
import numpy as np
import pandas as pd


class LabelEncoder:
    """
    Encodes pipe-separated finding strings (e.g., 'Cardiomegaly|Effusion' or 'No Finding')
    into 14-element multi-hot binary numpy vectors for PyTorch multi-label classification.
    """

    def __init__(self, expected_pathologies: List[str]):
        """
        Initialize the label encoder with the target pathology class list.

        Args:
            expected_pathologies (List[str]): List of 14 target pathology strings.
        """
        # Maintain a canonical sorted list of pathologies for consistent class indexing
        self.classes = list(expected_pathologies)
        self.num_classes = len(self.classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.idx_to_class = {i: name for i, name in enumerate(self.classes)}

    def encode(self, finding_label_str: str) -> np.ndarray:
        """
        Convert a pipe-separated string into a 14-element float32 multi-hot vector.

        Args:
            finding_label_str (str): Raw string from 'Finding Labels' column (e.g. 'Cardiomegaly|Effusion').

        Returns:
            np.ndarray: Multi-hot binary float32 array of shape (14,).
        """
        if not isinstance(finding_label_str, str) or not finding_label_str.strip():
            raise ValueError(f"Invalid or empty finding label string: '{finding_label_str}'")

        # Split pipe-separated string into individual label tokens
        tokens = [t.strip() for t in finding_label_str.split('|') if t.strip()]

        target_vector = np.zeros(self.num_classes, dtype=np.float32)

        for token in tokens:
            if token == "No Finding":
                # 'No Finding' is a normal scan indicator; target vector remains all zeros [0..0]
                continue
            elif token in self.class_to_idx:
                # Set 1.0 at the corresponding pathology class index
                idx = self.class_to_idx[token]
                target_vector[idx] = 1.0
            else:
                # Loud failure for unknown/unrecognized labels
                raise ValueError(
                    f"UNRECOGNIZED LABEL DETECTED: '{token}'. "
                    f"Expected one of {self.classes} or 'No Finding'."
                )

        return target_vector

    def decode(self, multi_hot_vector: Union[np.ndarray, List[float]], threshold: float = 0.5) -> List[str]:
        """
        Convert a 14-element multi-hot vector or probability vector back into string label names.

        Args:
            multi_hot_vector (np.ndarray): Array of shape (14,) with binary values or probabilities.
            threshold (float): Threshold above which a pathology is considered present.

        Returns:
            List[str]: List of predicted pathology string names, or ['No Finding'] if no pathology exceeds threshold.
        """
        arr = np.asarray(multi_hot_vector)
        predicted_indices = np.where(arr >= threshold)[0]

        if len(predicted_indices) == 0:
            return ["No Finding"]

        return [self.idx_to_class[idx] for idx in predicted_indices]

    def validate_dataframe(self, df: pd.DataFrame, label_column: str = "Finding Labels") -> None:
        """
        Validate every row in a DataFrame to verify all labels are valid.

        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            label_column (str): Name of the finding labels column.
        """
        print(f"[MediFusion] Validating '{label_column}' column across {len(df)} records...")
        
        all_raw_tokens = df[label_column].str.split('|').explode().str.strip()
        unique_tokens = set(all_raw_tokens.dropna().unique())

        valid_tokens = set(self.classes).union({"No Finding"})
        unexpected = unique_tokens - valid_tokens

        if unexpected:
            raise ValueError(
                f"VALIDATION FAILED! Found unexpected labels in metadata: {unexpected}. "
                f"Valid expected set is: {valid_tokens}"
            )

        print(f"[MediFusion] Label validation PASSED! All {len(unique_tokens)} unique tokens are valid.")
