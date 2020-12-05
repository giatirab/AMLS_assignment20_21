# Import Librariesis
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import sklearn

processing_available=[None, "crop", "crop&Canny"]

#class RandomForestClassifier(sklearn.ensemble.RandomForestClassifier):
#    
#    def __init__(self, canny = False, bodypart = None, resize = (72,72)):
#        self.canny = canny
#        self.bodypart = bodypart
#        self.resize = resize
#    
#    def process_dataset(self, X):
#        imgmgr = ImageManager()
#        imgmgr.images = X
#        if self.bodypart is not None:
#            imgmgr.bodypart = self.bodypart
#            imgmgr.crop_part(self.resize)
#        if self.canny:
#            imgmgr.canny_images()
#        X = imgmgr.images
#    
#    def fit(self, X_train, y_train):
#        return(sklearn.ensemble.RandomForestClassifier.fit(self, self.process_dataset(X_train), y_train))
#        
#    def predict(self, X_test):
#        return(sklearn.ensemble.RandomForestClassifier.predict(self, self.process_dataset(X_test)))
        


class ImageManager:

    CLASSIFIERS = {"face"  : "/Users/macbookpro/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml",
                   "smile"  : "/Users/macbookpro/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml",
                   #"smile" : "/Users/macbookpro/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_smile.xml",
                   "eyes"  : "/Users/macbookpro/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"}
    
    def __init__(self):
        self.images = np.array([])
        self.bodypart = None
        self.bodypart_classifier = None
    
    # currently not used
    def apply_to_images(func, *args):
        def new_function(self, *args):
            new_images = []
            for image in self.images:
                new_image = image.copy()
                processed_image = func(self, *args)(new_image)
                new_images.append(processed_image)
            self.images = np.array(new_images)
        return new_function
    
    def count_n_images(self, path, extension):
        list_dir = []
        list_dir = os.listdir(path)
        count = 0
        for file in list_dir:
            if file.endswith(extension):
                count += 1
        return count
    
    @property
    def images(self):
        return self._images
    
    @images.setter
    def images(self, new_images):
        if not(isinstance(new_images, type(np.array([])))):
            try:
                new_images = np.array(new_images)
            except:
                raise ValueError("The ImageManager's images are stored in numpy array format.")
        if(len(new_images)):
            self.image_size = new_images[0].shape
        else:
            self.image_size = 0
        new_images = np.asarray(new_images, np.uint8)
        self._images = new_images

    @property
    def bodypart(self):
        return self._bodypart
        
    @bodypart.setter
    def bodypart(self, new_bodypart):
        new_bodypart_classifier = None
        if new_bodypart is not None:
            if new_bodypart not in self.CLASSIFIERS.keys():
                raise ValueError("The bodypart {} is not supported. Check the keys of self.CLASSIFIERS!".format(new_bodypart))
            try:
                path = self.CLASSIFIERS[new_bodypart]
                new_bodypart_classifier = cv2.CascadeClassifier(path)
            except:
                raise ValueError("Was not able to create a cv2 classifier from the given filename\n{}".format(path))
        self._bodypart = new_bodypart
        self.bodypart_classifier = new_bodypart_classifier
    
    def load_images_from_folder(self, folder, extension, colour=cv2.COLOR_BGR2RGB, verbose=False):
        if verbose:
            print("Starting to load images from {} with extension {}".format(folder, extension))
            start_time = time.time()
        files = sorted(os.listdir(folder), key = lambda x: int(x[:-4]))
        n = self.count_n_images(folder, extension)
        im0 = np.asarray(cv2.imread(os.path.join(folder, files[0]), colour), np.uint8)
        shape = im0.shape
        X = np.empty((n, *shape), dtype=np.uint8)
        
        for i, image_name in enumerate(files):
            path = os.path.join(folder, image_name)
            image = np.asarray(cv2.imread(path, colour), np.uint8)
            if image.shape != shape:
                raise ValueError("All images with provided extension in the folder must have the same size.")
            X[i] = image
            if verbose:
                t = time.time()
                if t - start_time > 5 or i == n-1:
                    start_time = t
                    print("{}/{} images loaded".format(i+1, n))
        self.images = X
        self.folder_ = folder
        self.extension_ = extension
        return X
    
    def plot_image(self, image_number):
        image = self.images[image_number]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb, cmap = plt.cm.Spectral)
        plt.show()
        
    def canny_images(self, a = 100, b = 200):
        canny = []
        for image in self.images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canny.append(cv2.Canny(image, a, b))
        self.images = np.array(canny)
        return self.images
    
    def crop_part(self, resize_amt, squaring = False, x_offsets=(0,0), y_offsets = (0,0)):
        if self.bodypart_classifier is None:
            raise ValueError("You must set a body part before we can crop the images.")
        if len(self.images.shape) == 3:
            new_images = np.empty((len(self.images), *resize_amt))
        else:
            new_images = np.empty((len(self.images), *resize_amt, self.images.shape[-1]))
        
        def reduce(bboxes):
            size = 0
            i_max = 0
            for i, box in enumerate(bboxes):
                new_size = np.product(box.shape)
                if new_size > size:
                    size = new_size
                    i_max = i
            return bboxes[i_max]
        
        classifier = self.bodypart_classifier
        
        for i, image in enumerate(self.images):
            bboxes = classifier.detectMultiScale(image)
            if len(bboxes):   
                box = reduce(bboxes)
                x, y, width, height = box
                
                x_new = x + x_offsets[0]*width
                x2_new = x + width + x_offsets[1]*width
                y_new = y + y_offsets[0]*height
                y2_new = y + height + y_offsets[1]*height
                
                width_new, height_new = x2_new - x_new, y2_new - y_new
                if squaring: 
                    m = max(width_new, height_new)
                    width_new, height_new = m, m
                
                a = np.maximum(int(y_new), 0)
                b = np.minimum(int(y_new+height_new), image.shape[0]-1)
                c = np.maximum(int(x_new), 0)
                d = np.minimum(int(x_new+width_new), image.shape[1]-1)
                
                im = image[a:b, c:d] # slice di immagine iniziale
                new_img = im.copy()
                new_img = Image.fromarray(new_img)
            else: 
                new_img = Image.fromarray(image)
                
            new_img = new_img.resize(resize_amt)
            new_images[i] = np.asarray(new_img, np.uint8)
        self.images = np.array(new_images)
        return self.images
            
    def resize(self, resize_amt, images = None, setvalue = True):
        new_imges = []
        if not images:
            images = self.images
        for image in images:
            new_image = image.copy()
            new_image = new_image.resize(resize_amt, refcheck=True)
            new_imges.append(new_image)
        res = np.array(new_imges)
        if setvalue:
            self.images = res
        return np.array(new_imges)
        
        
#folder = "/Users/macbookpro/UCL - MSc Integrated Machine Learning Systems/Y1/Applied Machine Learning I/Final Assignment/AMLS_20-21_SN17024244/Datasets/dataset_AMLS_20-21/celeba/img/"
#extension = ".jpg"        
#imgmgr = ImageManager()
#imgmgr.load_images_from_folder(folder, extension, verbose = True)
#
#imgmgr.bodypart = "face"
##imgmgr.images= imgmgr.images[0:12]
#imges = imgmgr.crop_part((72,72))



