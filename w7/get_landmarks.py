import cv2
import numpy as np

import matplotlib.pyplot as plt
import torch
from datetime import datetime
from mobilenet import MobileNet


class FaceLandmark:
    def __init__(self,pretrained_model_path):
        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model= self.get_model(pretrained_model_path,self.device)
    
    def get_model(self,prefix,device):
        model=MobileNet(prefix).to(device)
        return model
    # lmk is prediction; src is template
    

    def get_landmark(self,imgs):
        """
        imgs: tensor images
        """
        image_size = (imgs.shape[-1], imgs.shape[-1])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.transpose(img, (2, 0, 1))  #3*112*112, RGB
        # img_tensor=torch.from_numpy(img)
        # img_tensor=img_tensor.unsqueeze(0)
        img_tensor=imgs.to(self.device)
        with torch.no_grad():
            preds=self.model(img_tensor)
        preds=preds.numpy()
        preds = preds.reshape((-1, 106,2))
        preds += 1
        preds *= (image_size[0] // 2)
        preds=np.round(preds).astype(np.int)
        # indexs=[1,17,2,4,6,0,22,20,18,52,61,71,53,54,57,63,67,56,59,77,80,83,86,72,73,74,35,39,33,40,89,93,87,94]
        indexs=[35,72,93]
        return preds[:,indexs,:]
if __name__ == '__main__':
    t1=datetime.now()
    prefix='/home/thanhdang/projects/insightface/alignment/model/kit_pytorch.npy' # path to pretrained model
    img_path1='/home/thanhdang/datasets/face_database/Aaron_Pena_0001.jpg'
    img_path2='/home/thanhdang/datasets/face_database/Abdel_Nasser_Assidi_0002.jpg'
    img_path3='/home/thanhdang/datasets/face_database/Abdulaziz_Kamilov_0001.jpg'
    img_path4='/home/thanhdang/datasets/face_database/AJ_Lamas_0001.jpg'

    faceLandmark=FaceLandmark(prefix)

    img_paths=[img_path1,img_path2,img_path3,img_path4]
    img_list=[]
    for i,img_path in enumerate(img_paths):
        img=cv2.imread(img_path)
        img=cv2.resize(img,(192,192))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=np.transpose(img,(2,0,1)) # 3x192x192
        img_list.append(img)
    imgs_input=np.array(img_list)
    imgs_tensor=torch.from_numpy(imgs_input)
    landmarks=faceLandmark.get_landmark(imgs_tensor)

    fig, ax = plt.subplots(1,4)
    for i,landmark in enumerate(landmarks):
        img=np.transpose(img_list[i],(1,2,0))
        # img_s = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i].imshow(img)
        ax[i].set_title("Original")
        tim1=img.copy()
        color = (255, 0, 0)
        for j in range(landmark.shape[0]):
            p = tuple(landmark[j])
            tim1 = cv2.circle(tim1, p, 2, color, -1)
            # tim1 = cv2.putText(tim1, str(
            #     j), (p[0],p[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
        ax[i].imshow(tim1)
        ax[i].set_title("Landmark")
    t2=datetime.now()
    print('Time: %s' %(t2-t1))
    plt.savefig('./result3.jpg')
    plt.show()
    cv2.waitKey(0)  