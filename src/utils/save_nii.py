import nibabel as nb

affine3D = [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]


def write_nii3D(data, file, voxel_size=None, offset=None):
    affine = affine3D.copy()
    if voxel_size is not None:
        affine[0][0], affine[1][1], affine[2][2] = voxel_size
    if offset is not None:
        affine[0][3], affine[1][3], affine[2][3] = offset

    nb.Nifti1Image(data, affine).to_filename(file)
