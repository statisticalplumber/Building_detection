import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
import cv2


class ProcessFile:
    def __init__(self, loc_image, loc_poly, orignal_image_size):
        self.loc_image = loc_image
        self.loc_poly = loc_poly
        self.image_poly_df = self.get_image_poly_dataframe()
        self.orignal_image_size = orignal_image_size

    def get_image_poly_dataframe(self):
        ind_img = os.listdir(self.loc_image)
        ind_poly = os.listdir(self.loc_poly)
        l_image = pd.DataFrame(ind_img, index=[i.split('.')[0] for i in ind_img])
        l_poly = pd.DataFrame(ind_poly, [i.split('.')[0] for i in ind_poly])
        d_img = pd.concat([l_image, l_poly], axis=1)
        d_img = d_img.dropna()
        d_img.columns = ['image', 'poly']
        return d_img

    def get_custom_image_mask(self, size=[128,128]):
        d_img = self.image_poly_df
        l_image_file = []
        l_polygon = []
        for count, img, poly in zip(range(len(d_img)), d_img.image, d_img.poly):
            try:
                m = image.img_to_array(image.load_img(self.loc_image + img, grayscale=True).resize(size))
                l_image_file.append(m.reshape([len(m) ** 2]))
                dp = pd.read_table(self.loc_poly, header=None)
                xx = [eval(i) for i in dp.iloc[:, 0].tolist()]
                im = np.zeros([len(m)] * 2, dtype=np.uint8)
                for i in xx:
                    a3 = np.array([i], dtype=np.int32) //(self.orignal_image_size/len(m))
                    im = cv2.fillPoly(im, a3, 255)
                l_polygon.append(im.reshape([len(m) ** 2]))
            except:
                pass
        df_image = pd.DataFrame(l_image_file)
        df_mask = pd.DataFrame(l_polygon)
        return df_image, df_mask

if __name__=="__main__":

    loc_image = './data/images/'
    loc_poly = './data/polygon/'
    pfile = ProcessFile(loc_image, loc_poly, 1280)
    processed_image_df, mask_df = pfile.get_custom_image_mask(size=[128, 128])

    processed_image_df.to_csv('./data/train/mask_file.csv', index=False)
    mask_df.to_csv('./data/train/train_file.csv', index=False)

