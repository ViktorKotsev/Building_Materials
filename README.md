After installing the requirements.txt, to produce a segmentation mask run: "python predict.py -i [Path_to_Folder_containing_img_or_vk4_files] -o [Path_to_Output_folder] -m [Path_to_Model_checkpoint] -c 3 -s 0.2".

Note that it is important to control the size of the input images through '-s' if the resolution is different from the original 2048/1756. For example, if the resolution is half of the original, then the size should be doubled to -s 0.4. Otherwise, the performance will be severely affected.
