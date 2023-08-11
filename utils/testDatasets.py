import sys
# set your codes base folder
sys.path.append('../Tensorflow-Unet-Brain-Tumor-Segmentation')
from config import TRAIN_DATASET_PATH, VALIDATION_DATASET_PATH
import matplotlib.pyplot as plt
import nibabel as nib

train_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_005/BraTS20_Training_005_flair.nii').get_fdata()
train_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_005/BraTS20_Training_005_t1.nii').get_fdata()
train_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_005/BraTS20_Training_005_t1ce.nii').get_fdata()
train_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_005/BraTS20_Training_005_t2.nii').get_fdata()
train_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_005/BraTS20_Training_005_seg.nii').get_fdata()

valid_image_flair=nib.load(VALIDATION_DATASET_PATH + 'BraTS20_Validation_040/BraTS20_Validation_040_flair.nii').get_fdata()
valid_image_t1=nib.load(VALIDATION_DATASET_PATH + 'BraTS20_Validation_040/BraTS20_Validation_040_t1.nii').get_fdata()
valid_image_t1ce=nib.load(VALIDATION_DATASET_PATH + 'BraTS20_Validation_040/BraTS20_Validation_040_t1ce.nii').get_fdata()
valid_image_t2=nib.load(VALIDATION_DATASET_PATH + 'BraTS20_Validation_040/BraTS20_Validation_040_t2.nii').get_fdata()

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
fig.canvas.manager.set_window_title('Training')
slice_w = 25
ax1.imshow(train_image_flair[:,:,train_image_flair.shape[0]//2-slice_w], cmap = 'gray')
ax1.set_title('Train Image flair')
ax2.imshow(train_image_t1[:,:,train_image_t1.shape[0]//2-slice_w], cmap = 'gray')
ax2.set_title('Train Image t1')
ax3.imshow(train_image_t1ce[:,:,train_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
ax3.set_title('Train Image t1ce')
ax4.imshow(train_image_t2[:,:,train_image_t2.shape[0]//2-slice_w], cmap = 'gray')
ax4.set_title('Train Image t2')
ax5.imshow(train_mask[:,:,train_mask.shape[0]//2-slice_w])
ax5.set_title('Train Mask')

plt.show()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (20, 10))
fig.canvas.manager.set_window_title('Validation')
slice_w = 25
ax1.imshow(valid_image_flair[:,:,valid_image_flair.shape[0]//2-slice_w], cmap = 'gray')
ax1.set_title('Valid Image flair')
ax2.imshow(valid_image_t1[:,:,valid_image_t1.shape[0]//2-slice_w], cmap = 'gray')
ax2.set_title('Valid Image t1')
ax3.imshow(valid_image_t1ce[:,:,valid_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
ax3.set_title('Valid Image t1ce')
ax4.imshow(valid_image_t2[:,:,valid_image_t2.shape[0]//2-slice_w], cmap = 'gray')
ax4.set_title('Valid Image t2')

plt.show()