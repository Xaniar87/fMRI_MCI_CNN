import os
import numpy as np
# import logging
from sklearn.model_selection import train_test_split
import pydicom
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn_2layer import cnn_2layer
from config import set_args

# args = set_args()
# # set model dir
# model_dir = args.model_dir
# # setup logger
# logger =  create_logger(__name__, to_disk=True, log_file=args.log_file)


height=128
width=128
channel=1
n_inputs = channel*height*width
n_fc1 = 64
n_outputs = 3
# sess2=tf.Session()
# path="C:/Users/Xaniar/Anaconda3/Scripts/MyProjects/ImageProc/fMRI/images/"
path="./images/"
images=[]
classes=[]
for k in range (0,54):
    files_names=os.listdir(path)
#     print(k)
    ds = pydicom.dcmread("images/"+files_names[k])
    data = ds.pixel_array
#     d=data[700:730,700:730]
#     plt.imshow(d, cmap=plt.cm.bone)
#     plt.show()
    for i in range (0, 8):
        for j in range (0,8):
            image=data[(i*128):(i*128)+128,(j*128):(j*128)+128]
            image=image.flatten()
            images.append(image)
            classname=files_names[k].split(".")[0]
            if classname=="MCI1":
                lbl=0
            elif classname=="HY01":
                lbl=1
            else:
                lbl=2
            classes.append(lbl)
train_images, test_images, train_labels, test_labels = train_test_split(images, classes, test_size=0.30)
tf.reset_default_graph()
logits, Y_proba, X=cnn_2layer (n_inputs, n_outputs, height, width, channel)

with tf.name_scope("train"):
    y = tf.placeholder(tf.int32, shape = [None], name = "y")
#     print (logits,y)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

n_epochs = 1
batch_size = 64
train_mode = tf.placeholder(tf.bool)
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        print (epoch)
        for iteration in range(batch_size):
            idx = np.random.choice(np.where(train_images)[0], size= batch_size )
#             print(idx)
#             X_batch =list(train_images[i] for i in idx )
            X_batch = [train_images[x] for x in idx]        
#             X_batch = train_images[idx,:]
            y_batch = [train_labels[x] for x in idx]
            y_batch = np.reshape(y_batch,batch_size)
            len(X_batch)
            len(y_batch)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch,train_mode: 1})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch,train_mode: 0})
        acc_test = accuracy.eval(feed_dict={X: test_images, y: test_labels,train_mode: 0})
        print("Epoch:",epoch+1, "Train accuracy:", acc_train, "Validation accuracy:", acc_test)
        # logger.warning("Epoch {0} - test ACC: {1:.4f}".format(epoch+1, acc_test))
