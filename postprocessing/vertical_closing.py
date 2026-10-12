import os
import glob
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from PIL import Image

# ==========================================
# 1. CORE POSTPROCESSING FUNCTION (From before)
# ==========================================
def close_esophagus_vertical(volume_mask, esophagus_label=4, z_radius=5, xy_radius=1):
    eso_binary = (volume_mask == esophagus_label)
    z_len = 2 * z_radius + 1
    xy_len = 2 * xy_radius + 1
    
    footprint = np.zeros((z_len, xy_len, xy_len), dtype=bool)
    footprint[:, xy_radius, xy_radius] = True
    
    if xy_radius > 0:
        footprint[z_radius, :, xy_radius] = True
        footprint[z_radius, xy_radius, :] = True

    closed_eso = ndimage.binary_closing(eso_binary, structure=footprint)
    
    refined_volume = volume_mask.copy()
    refined_volume[volume_mask == esophagus_label] = 0
    refined_volume[closed_eso] = esophagus_label
    
    return refined_volume


# ==========================================
# 2. PIPELINE FOR 3D UNETR (NIfTI File)
# ==========================================
def postprocess_nifti(input_nii_path, output_nii_path, esophagus_label=4):
    # Load NIfTI file
    nii_img = nib.load(input_nii_path)
    # Get 3D array data. Shape is usually (X, Y, Z) in NiBabel natively
    volume_data = nii_img.get_fdata().astype(np.int16)
    
    # Crucial: Swap axes if your vertical axis isn't the first dimension in the function.
    # The function expects Z (slices) as the first dimension: (Z, Y, X)
    # If get_fdata() gives you (X, Y, Z), transpose it:
    volume_data_zyx = np.transpose(volume_data, (2, 1, 0))
    
    # Apply vertical morphological closing
    refined_volume_zyx = close_esophagus_vertical(volume_data_zyx, esophagus_label)
    
    # Transpose back to original NIfTI orientation (X, Y, Z)
    refined_volume_xyz = np.transpose(refined_volume_zyx, (2, 1, 0))
    
    # Save back as NIfTI preserving original metadata (affine matrix)
    new_nii = nib.Nifti1Image(refined_volume_xyz, nii_img.affine, nii_img.header)
    nib.save(new_nii, output_nii_path)
    print(f"Processed NIfTI saved to: {output_nii_path}")


# ==========================================
# 3. PIPELINE FOR 2D ENet (PNG Slices Folder)
# ==========================================
def postprocess_png_slices(png_folder_path, output_folder_path, esophagus_label=4):
    # Find all PNG prediction slices. Make sure they are sorted in vertical order (001.png, 002.png...)
    png_paths = sorted(glob.glob(os.path.join(png_folder_path, "*.png")))
    
    if not png_paths:
        print("No PNG files found in the specified folder.")
        return

    # Load and stack all 2D slices into a 3D volume along the Z axis
    slices_list = []
    for path in png_paths:
        img = Image.open(path)
        slices_list.append(np.array(img))
    
    # Shape becomes (Z, Y, X) where Z is the number of slices
    volume_data_zyx = np.stack(slices_list, axis=0).astype(np.int16)
    
    # Apply vertical morphological closing
    refined_volume_zyx = close_esophagus_vertical(volume_data_zyx, esophagus_label)
    
    # Unstack the 3D volume back into individual 2D PNG slices
    os.makedirs(output_folder_path, exist_ok=True)
    for z in range(refined_volume_zyx.shape[0]):
        # Extract the slice and convert back to image format
        slice_2d = refined_volume_zyx[z].astype(np.uint8)
        img_out = Image.fromarray(slice_2d)
        
        # Save matching the original filename
        original_filename = os.path.basename(png_paths[z])
        img_out.save(os.path.join(output_folder_path, original_filename))
        
    print(f"Processed PNG slices saved to: {output_folder_path}")