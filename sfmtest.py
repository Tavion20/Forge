import pycolmap
import os

def run_sfm(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize COLMAP and set paths
    database_path = os.path.join(output_dir, 'database.db')
    sparse_dir = os.path.join(output_dir, 'sparse')
    dense_dir = os.path.join(output_dir, 'dense')

    # Feature Extraction
    pycolmap.extract_features(image_dir=image_dir, database_path=database_path)
    
    # Feature Matching
    pycolmap.match_exhaustive(database_path=database_path)
    
    # Sparse Reconstruction (Mapping)
    pycolmap.sfm(database_path=database_path, image_dir=image_dir, output_path=sparse_dir)

    # Dense Reconstruction
    pycolmap.dense_reconstruction(
        image_dir=image_dir, 
        sparse_model_dir=os.path.join(sparse_dir, '0'),  # Assuming model is in subdir '0'
        output_dir=dense_dir
    )

# Usage example
image_dir = './extracted/'  # Folder containing extracted video frames
output_dir = './colmap_output/'
run_sfm(image_dir, output_dir)
