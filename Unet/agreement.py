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
    assert len(folder_paths) == 5, "Exactly 7 folders must be provided"

    # Create the destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Get the set of common mask files present in all seven folders
    common_files = set(os.listdir(folder_paths[0])).intersection(*[set(os.listdir(folder)) for folder in folder_paths[1:]])

    if not common_files:
        print("No common mask files found in all seven folders.")
        return


    for filename in common_files:
        mask_paths = [os.path.join(folder, filename) for folder in folder_paths]

        # Load the masks
        masks = [load_mask(path) for path in mask_paths]

        # Calculate the agreement map and majority voting map
        agreement_map, majority_vote = calculate_agreement(*masks)

        # Calculate the overall agreement score
        score = overall_agreement_score(agreement_map)

        # Output the results
        print(f"File: {filename}, Overall Agreement Score: {score:.2f}")

        # Save
        if score > 0.97:
            # Save the majority voting map as an image
            majority_vote_image = Image.fromarray(majority_vote)
            majority_vote_image_path = os.path.join(dest_folder, f"{filename}")
            majority_vote_image.save(majority_vote_image_path)
            print(f"Majority vote map saved as '{majority_vote_image_path}'")


if __name__ == "__main__":
    # Example usage with three mask file paths
    mask_paths = ["/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/focal/M1S1/M1S1_1_P_U1-GR2020", 
    "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/height/M1S1/M1S1_1_P_U1-GR2020", 
    "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/height_focal/M1S1/M1S1_1_P_U1-GR2020",
    "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/light_height/M1S1/M1S1_1_P_U1-GR2020",
    "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/wd/M1S1/M1S1_1_P_U1-GR2020"  ]

    output_folder = "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/new_data/M1S1/M1S1_1_P_U1-GR2020/"
    main(mask_paths, output_folder)