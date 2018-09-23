import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

def prepare():
    imgdata=[]
    classdata=[]
    labels=np.loadtxt("../bird identification/CUB_200_2011/image_class_labels.txt",dtype=str)
    labels=labels[:,1]
    image_path=np.loadtxt("../bird identification/CUB_200_2011/classes.txt",dtype=str)
    image_path=image_path[:,1]
    birds=np.loadtxt("../bird identification/CUB_200_2011/images.txt",dtype=str)
    birds=birds[:,1]
    print(birds[0])
    for images in birds:
        filename="../bird identification/CUB_200_2011/images/"+str(images)
        img=image.load_img(filename,target_size=(200,200))
        x1=image.img_to_array(img)
        print(filename)
        imgdata.append(x1)
    image_data=np.array(imgdata)
    print(image_data.shape)
    print(labels.shape)
    np.save('image_data.npy',image_data)
    

prepare()
