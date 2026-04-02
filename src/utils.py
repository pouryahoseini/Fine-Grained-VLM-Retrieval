import scipy.io as sio
import os
from typing import Tuple

def parse_stanford_mat_files(data_dir: str, split: str = "train") -> Tuple[list, list]:
    """
    Parses the legacy MATLAB annotation files into a flat list of dictionaries.
    Assumes the dataset is structured like:
    data_dir/
      ├── cars_meta.mat
      ├── cars_train_annos.mat
      ├── cars_test_annos_withlabels.mat
      ├── cars_train/  (Folder of train JPEGs)
      └── cars_test/   (Folder of test JPEGs)
    Args:
        data_dir (str): The root directory of the Stanford Cars dataset.
        split (str): "train" or "test" to specify which annotations to parse.
    Returns:
        records (list): A list of dictionaries, each containing:
            - "image_path": Full path to the image file.
            - "label_name": The class name (e.g., "2010 Ford F-150").
            - "bbox": A dictionary with keys "x1", "y1", "x2", "y2" for the bounding box coordinates.
        class_names (list): A list of all class names corresponding to the class IDs in the annotations.
    """
    # Extract the class names
    meta_path = os.path.join(data_dir, "cars_meta.mat")
    meta_data = sio.loadmat(meta_path)
    
    # Get class names
    class_names = [c[0] for c in meta_data['class_names'][0]]

    # Extract the annotations based on the split
    if split == "train":
        anno_path = os.path.join(data_dir, "cars_train_annos.mat")
        img_folder = os.path.join(data_dir, "cars_train")
    else:
        anno_path = os.path.join(data_dir, "cars_test_annos_withlabels.mat")
        img_folder = os.path.join(data_dir, "cars_test")

    anno_data = sio.loadmat(anno_path)
    annotations = anno_data['annotations'][0]

    records = []
    
    # Build the clean dictionary
    for anno in annotations:
        # Extract the filename string from the nested array
        fname = anno['fname'][0]
        image_path = os.path.join(img_folder, fname)
        
        # MATLAB uses 1-based indexing. Python uses 0-based
        class_id = int(anno['class'][0][0]) - 1
        label_name = class_names[class_id]
        
        # Extracting the bounding box
        bbox = {
            "x1": int(anno['bbox_x1'][0][0]),
            "y1": int(anno['bbox_y1'][0][0]),
            "x2": int(anno['bbox_x2'][0][0]),
            "y2": int(anno['bbox_y2'][0][0]),
        }

        records.append({
            "image_path": image_path,
            "label_name": label_name,
            "class_id": class_id,
            "bbox": bbox
        })

    return records, class_names