import os
import sys
import cv2
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# set your codes base folder
sys.path.append('../Tensorflow-Unet-Brain-Tumor-Segmentation')
from config import VOLUME_START, VOLUME_SLICES, IMG_SIZE, TRAIN_DATASET_PATH

#======================================================

SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE',
    2 : 'EDEMA',
    3 : 'ENHANCING'
}

#======================================================

# to run this section first : sudo apt install graphviz -y 
# from tensorflow.keras.utils import plot_model
# plot_model(model, 
#            show_shapes = True,
#            show_dtype=False,
#            show_layer_names = True, 
#            rankdir = 'TB', 
#            expand_nested = False, 
#            dpi = 70)

#======================================================

def trainingResults(hist, saveName):
    
    acc=hist['accuracy']
    val_acc=hist['val_accuracy']

    epoch=range(len(acc))

    loss=hist['loss']
    val_loss=hist['val_loss']

    train_dice=hist['dice_coef']
    val_dice=hist['val_dice_coef']

    fig,ax=plt.subplots(1,4,figsize=(16,8))
    fig.canvas.manager.set_window_title('Training Results')

    ax[0].plot(epoch,acc,'g',label='Training Accuracy')
    ax[0].plot(epoch,val_acc,'b',label='Validation Accuracy')
    ax[0].legend()

    ax[1].plot(epoch,loss,'b',label='Training Loss')
    ax[1].plot(epoch,val_loss,'r',label='Validation Loss')
    ax[1].legend()

    ax[2].plot(epoch,train_dice,'b',label='Training dice coef')
    ax[2].plot(epoch,val_dice,'r',label='Validation dice coef')
    ax[2].legend()

    ax[3].plot(epoch,hist['mean_io_u'],'b',label='Training mean IOU')
    ax[3].plot(epoch,hist['val_mean_io_u'],'r',label='Validation mean IOU')
    ax[3].legend()

    plt.savefig(f"{saveName}.png")
    plt.show()

#======================================================

class prediction:

    def __init__(self, model, test_ids, caseNumber=1, start_slices=50) -> None:
        self.model = model
        self.test_ids = test_ids
        self.caseNumber = caseNumber
        self.slices = start_slices
        self.displayPredictsById()
        
    def predictByPath(self, case_path,case):
        next(os.walk(case_path))[2]
        X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    #  y = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE))
        
        vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii')
        flair=nib.load(vol_path).get_fdata()
        
        vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii')
        ce=nib.load(vol_path).get_fdata() 
        
    #   vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_seg.nii');
    #   seg=nib.load(vol_path).get_fdata()  

        
        for j in range(VOLUME_SLICES):
            X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START], (IMG_SIZE,IMG_SIZE))
            X[j,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START], (IMG_SIZE,IMG_SIZE))
    #       y[j,:,:] = cv2.resize(seg[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
            
        return X


    def displayPredictsById(self):
        for i in range (self.caseNumber) :
            path = TRAIN_DATASET_PATH + f"BraTS20_Training_{self.test_ids[i][-3:]}"
            gt = nib.load(os.path.join(path, f'BraTS20_Training_{self.test_ids[i][-3:]}_seg.nii')).get_fdata()
            origImage = nib.load(os.path.join(path, f'BraTS20_Training_{self.test_ids[i][-3:]}_flair.nii')).get_fdata()
            matX = self.predictByPath(path, self.test_ids[i][-3:])
            p = self.model.predict(matX/np.max(matX), verbose=1)

            core = p[:,:,:,1]
            edema= p[:,:,:,2]
            enhancing = p[:,:,:,3]

            # plt.figure(figsize=(10, 30))
            fig, axarr = plt.subplots(1, 6, figsize=(18, 50)) 
            fig.canvas.manager.set_window_title('prediction')

            for i in range(6):
                axarr[i].clear()
                axarr[i].imshow(cv2.resize(origImage[:,:,self.slices+VOLUME_START], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')

            axarr[0].imshow(cv2.resize(origImage[:,:,self.slices+VOLUME_START], (IMG_SIZE, IMG_SIZE)), cmap="gray")
            axarr[0].title.set_text('Original image flair')
            curr_gt=cv2.resize(gt[:,:,self.slices+VOLUME_START], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
            
            axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
            axarr[1].title.set_text('Ground truth')
            
            axarr[2].imshow(p[self.slices,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
            axarr[2].title.set_text('all classes')
            
            axarr[3].imshow(edema[self.slices,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
            axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
            
            axarr[4].imshow(core[self.slices,:,], cmap="OrRd", interpolation='none', alpha=0.3)
            axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
            
            axarr[5].imshow(enhancing[self.slices,:,], cmap="OrRd", interpolation='none', alpha=0.3)
            axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')

            plt.savefig(f"predictResults/BraTS20_Training_{self.test_ids[i][-3:]}.png")
            plt.show()

# mask = np.zeros((10,10))
# mask[3:-3, 3:-3] = 1 # white square in black background
# im = mask + np.random.randn(10,10) * 0.01 # random image
# masked = np.ma.masked_where(mask == 0, mask)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(im, 'gray', interpolation='none')
# plt.subplot(1,2,2)
# plt.imshow(im, 'gray', interpolation='none')
# plt.imshow(masked, 'jet', interpolation='none', alpha=0.7)
# plt.show()