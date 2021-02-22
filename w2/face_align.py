import cv2
import numpy as np
from skimage import transform as trans
import mxnet as mx
import matplotlib.pyplot as plt
from datetime import datetime

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

# arcface_src = np.expand_dims(arcface_src, axis=0)

# In[66]:

class FaceAlign:
    def __init__(self,pretrained_model_path,epoch):
        self.model= self.get_model(pretrained_model_path,epoch,image_size=192)
    
    def align(self,image_path):
        img = cv2.imread(img_path)
        img=cv2.resize(img,(192,192))
        landmarks=self.get_landmark(img,self.model)
        landmarks = np.round(landmarks).astype(np.int)
        indexs=[38,88,74,52,68]
        wraped=self.norm_crop(img,landmarks[indexs],image_size=112,mode="arcface")
        return wraped
    def norm_crop(self,img, landmarks, image_size=192, mode='arcface'):
        M, pose_index = self.estimate_norm(landmarks, image_size, mode)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

    def estimate_norm(self,lmk, image_size=112, mode='arcface'):
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        if mode == 'arcface':
            assert image_size == 112
            src = np.expand_dims(src2, axis=0)
        else:
            src = src_map[image_size]
        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
            #         print(error)
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index
    def get_model(self,prefix,epoch,image_size):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
        model.bind(for_training=False,
                    data_shapes=[('data', (1, 3, image_size, image_size))
                                ])
        model.set_params(arg_params, aux_params)
        return model
    # lmk is prediction; src is template
    

    def get_landmark(self,img,model):
        image_size = (img.shape[1], img.shape[1])
        input_blob = np.zeros((1, 3) + image_size, dtype=np.float32)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  #3*112*112, RGB
        input_blob[0] = img
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        model.forward(db, is_train=False)
        pred = model.get_outputs()[-1].asnumpy()[0]
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (image_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (image_size[0] // 2)
        return pred
if __name__ == '__main__':
    t1=datetime.now()
    prefix='C:\\Users\\thanhdh6\\Documents\\projects\\insightface\\alignment\\model\\2d106det' # path to pretrained model
    img_path1='C:\\Users\\thanhdh6\\Downloads\\face-2-289x300.png'
    img_path2='C:\\Users\\thanhdh6\\Downloads\\argan-oil-for-your-face_1920x1080.jpg'
    img_path3='C:\\Users\\thanhdh6\\Documents\\datasets\\lfw\\Abid_Hamid_Mahmud_Al-Tikriti\\Abid_Hamid_Mahmud_Al-Tikriti_0002.jpg'
    
    fig, ax = plt.subplots(3,2)
    faceAlign=FaceAlign(prefix,0)

    img_paths=[img_path1,img_path2,img_path3]
    for i,img_path in enumerate(img_paths):
        wraped = faceAlign.align(img_path)
        img = cv2.imread(img_path)
        img=cv2.resize(img,(192,192))
        img_s = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i][0].imshow(img_s)
        ax[i][0].set_title("Original")
        wraped_s = cv2.cvtColor(wraped, cv2.COLOR_BGR2RGB)
        ax[i][1].imshow(wraped_s)
        ax[i][1].set_title("Aligned")
    t2=datetime.now()
    print('Time: %s' %(t2-t1))
    plt.show()
    cv2.waitKey(0)  

