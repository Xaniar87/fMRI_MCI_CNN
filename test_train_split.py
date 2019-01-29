def test_train_split (dir, height, width,channel):
    import tensorflow as tf
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as mt
    from sklearn.model_selection import train_test_split
    import test_train_split

    sess2=tf.Session()
    def read_file_content(path, file_name):
        """ read content of the given png file"""
        file_contents = tf.read_file(path+"/"+ file_name)
        f = tf.image.decode_png(file_contents, channels=channel)
        f = tf.image.resize_images(f, [height, width])
        d=f.eval(session=sess2)
        return d.ravel()


    # dir="D:/Multi-Pie/data"
    # dir="C:/Users/Xaniar/Anaconda3/Scripts/MyProjects/ImageProc/Multi-Pie/data/"
    content=os.listdir(dir)
    Imageslist = []
    LabelList = []
    images=[np.zeros(height*width*channel)]
    # print(content)
    for i in content:
        dir1=dir+i+"/"+"multiview3"
        content1=os.listdir(dir1)
    #     print(content1)
        for j in content1:
          dir2=dir1+"/"+j
          content2=os.listdir(dir2)
    #       print(content2)
          for k in content2:
            dir3=dir2+"/"+k
            content3=os.listdir(dir3)
    #         print(content3)
            for l in content3:
                dir4=dir3+"/"+l
                content4=os.listdir(dir4)
                Imageslist=np.append(Imageslist,content4)
    #             images=np.array([read_file_content(dir4,Imageslist[0])])
    #             images=images.tolist
                for s in range(len(content4)):
                    lbl=np.array(content4[s].split("_")[0])
    #                 labels=np.array([int(x.split("_")[0]) for x in files_names])-1
                    LabelList = np.append(LabelList,lbl)
                    img=[read_file_content(dir4,Imageslist[-1])]
                    images=np.concatenate((images,img), axis=0)
    #                 img=img.tolist
    #                 print(img)
    #                 print(images)
                    
    #                 images=np.concatenate((images,img), axis=0)
    #                 images=np.array([read_file_content(dir4,x) for x in Imageslist])
    #             print(dir4)

                    
                

    images=np.reshape(images, (len(LabelList)+1,height*width*channel))
    # images=np.delete(images,[0:height*width*channel])
    images=images[1:]
    LabelList = list(map(int, LabelList))
    LabelList[:] = [x - 1 for x in LabelList]
    # print (LabelList)
    # print(sum(LabelList))
    train_images, test_images, train_labels, test_labels = train_test_split(images, LabelList, test_size=0.20)

    return train_images, test_images, train_labels, test_labels


