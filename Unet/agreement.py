import os
import numpy as np
from PIL import Image

def load_mask(file_path):
    """Load the prediction mask from a .png file and convert it to a numpy array."""
    return np.array(Image.open(file_path))

def calculate_agreement(*masks):
    """Calculate the agreement between multiple prediction masks and produce a majority voting map."""
    # Ensure the masks are of the same shape
    shapes = [mask.shape for mask in masks]
    assert all(shape == shapes[0] for shape in shapes), "All masks must have the same shape"

    num_masks = len(masks)
    mask_shape = masks[0].shape

    # Initialize agreement map and majority voting map
    agreement = np.zeros(mask_shape, dtype=np.float32)
    majority_vote = np.zeros(mask_shape, dtype=masks[0].dtype)

    # Calculate agreement and majority voting
    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            votes = [mask[i, j] for mask in masks]
            vote_counts = {vote: votes.count(vote) for vote in set(votes)}
            if len(vote_counts) == 3:
                max_count = 0
            else:
                max_count = max(vote_counts.values())
            majority_candidates = [vote for vote, count in vote_counts.items() if count == max_count]

            if len(majority_candidates) == 1:
                majority_vote[i, j] = majority_candidates[0]
            else:
                majority_vote[i, j] = votes[0]  # Tie-breaking by selecting the value from the first model

            agreement_score = max_count / num_masks
            agreement[i, j] = agreement_score

    return agreement, majority_vote

def overall_agreement_score(agreement):
    """Calculate the overall agreement score."""
    return np.mean(agreement)

def main(folder_paths, dest_folder):
    assert len(folder_paths) == 5, "Exactly 5 folders must be provided"

    # Create the destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Collect relative paths of all image files in each folder
    relative_image_paths_list = []
    for folder in folder_paths:
        relative_image_paths = set()
        for dirpath, _, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith('.png'):  # Adjust the extension if needed
                    abs_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(abs_path, folder)
                    relative_image_paths.add(rel_path)
        relative_image_paths_list.append(relative_image_paths)

    # Find the intersection of relative image paths across all five folders
    common_relative_paths = set.intersection(*relative_image_paths_list)

    if not common_relative_paths:
        print("No common image files found in all five folders.")
        return

    for rel_path in common_relative_paths:
        mask_paths = [os.path.join(folder, rel_path) for folder in folder_paths]

        # Load the masks
        masks = [load_mask(path) for path in mask_paths]

        # Calculate the agreement map and majority voting map
        agreement_map, majority_vote = calculate_agreement(*masks)

        # Calculate the overall agreement score
        score = overall_agreement_score(agreement_map)

        # Output the results
        print(f"File: {rel_path}, Overall Agreement Score: {score:.2f}")

        # Save if the score is above the threshold
        if score > 0.97:
            # Ensure the directory exists in dest_folder
            output_path = os.path.join(dest_folder, rel_path)
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            # Save the majority voting map as an image
            majority_vote_image = Image.fromarray(majority_vote)
            majority_vote_image.save(output_path)
            print(f"Majority vote map saved as '{output_path}'")

if __name__ == "__main__":
    # Example usage with five mask folder paths
    mask_paths = [
        "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/New/250_flip",
        "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/New/250_color",
        "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/New/250_crop",
        "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/New/250_crop_color",
        "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/New/250"
    ]

    output_folder = "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/ensemble_97"
    main(mask_paths, output_folder)