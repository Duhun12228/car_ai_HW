import cv2 as cv
import numpy as np
import os

class data_argumentation:
    def __init__(self,img_list):
        self.data = img_list
    
    def blur(self,filter_size):
        if self.data is None:
            return
        for i in range(len(self.data)):
            self.data[i] = cv.blur(self.data[i],(filter_size,filter_size))
    
    def guassian_noise(self,mean=0,std=1):
        if self.data is None:
            return
        for i in range(len(self.data)):
            noise = np.random.normal(mean,std,self.data[i].shape)
            self.data[i] = self.data[i] + noise.astype(np.uint8)
    
    def translation(self,tx,ty):
        if self.data is None:
            return
        for i in range(len(self.data)):
            rows,cols = self.data[i].shape[:2]
            M = np.float32([[1,0,tx],[0,1,ty]])
            dst = cv.warpAffine(self.data[i],M,(cols,rows))
            self.data[i] = dst
        
    def rotation(self,angle):
        if self.data is None:
            return
        for i in range(len(self.data)):
            rows,cols = self.data[i].shape[:2]
            M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
            dst = cv.warpAffine(self.data[i],M,(cols,rows))
            self.data[i] = dst
    
    def total_argumentation(self,filter_size,std,tx,ty,angle):
        self.blur(filter_size)
        self.guassian_noise(std=std)
        self.translation(tx,ty)
        self.rotation(angle)
    
    def save_images(self,index):
        if self.data is None:
            return
        
        if not os.path.exists(f'result_{index}'):
            os.makedirs(f'result_{index}')
        
        print(f'Saving images to result_{index}....')

        for i in range(len(self.data)):
            cv.imwrite(f'result_{index}/argumented_image{i}.jpg',self.data[i])
            print(f'result_{index}/argumented_image{i} is saved.')
            cv.waitKey(0)



if __name__ == "__main__":
    name_list = [x for x in range(0,10)]
    h_list = ['du','eo','ga','geo','go','gu','heo','jeo','jo','ju']
    file_list = name_list + h_list
    img_list = []
    for file in file_list:
        img_list.append(cv.imread(f'{file}.jpg'))
    count = 0

    for filter_size in [3,5]:
        for noise_std in range(1,21,2):
            for tx,ty in enumerate(range(1,11)): 
                tx += 1
                for angle in range(-5,6,1):
                    count += 1
                    copied_list = [img.copy() for img in img_list]
                    data_arg = data_argumentation(copied_list)
                    data_arg.total_argumentation(filter_size,noise_std,tx,ty,angle)
                    data_arg.save_images(count)




        
