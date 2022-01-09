import numpy as np
import pandas as pd
import sklearn.datasets import fetch_openml
from sklearn.model_selection import trian_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
X,y=fetch_openml("mnist_784",version=1,return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=9,train_size=7500,test_size=50)
X_train_scaled=X_train/255
X_test_scaled=X_test/255
clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_scaled,y_train)
def getprediction(image):
    im_PIL=Image.open(image)
    image_bw=im_PIL.convert("L")
    image_bw_resized=image_bw.resized((28,28),Image.ANTIALIAS)
    pixel_filter=20
    minpixel=np.percentile(image_bw_resized,pixel_filter)
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized-minpixel,0,255)
    maxpixel=np.max(image_bw_resized)
    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/maxpixel
    testsample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    testpred=clf.predict(testsample)
    return testpred[0]