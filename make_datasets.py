import numpy as np
from PIL import Image
# import utility as Utility
import os
import csv
import random
import utility as util

class Make_datasets_Food101():
    def __init__(self, base_dir, img_width, img_height, image_dirX, image_dirY):

        self.base_dir = base_dir
        self.img_width = img_width
        self.img_height = img_height
        # self.list_train_files = []
        self.dirX = base_dir + image_dirX
        self.dirY = base_dir + image_dirY

        self.file_listX = os.listdir(self.dirX)
        self.file_listY = os.listdir(self.dirY)

        self.image_fileX_num = len(self.file_listX)
        self.image_fileY_num = len(self.file_listY)
        print("self.img_width", self.img_width)
        print("self.img_height", self.img_height)
        print("len(self.file_list1)", len(self.file_listX))
        print("len(self.file_list2)", len(self.file_listY))
        print("self.image_file1_num", self.image_fileX_num)
        print("self.image_file2_num", self.image_fileY_num)


    def read_1_data(self, dir, filename_list, width, height):
        images = []
        for filename in filename_list:
            pilIn = Image.open(dir + filename)
            pilResize = pilIn.resize((width, height))
            image = np.asarray(pilResize, dtype=np.float32)
            try:
                image_t = np.transpose(image, (2, 0, 1))
            except:
                print("filename =", filename)
                image_t = image.reshape(image.shape[0], image.shape[1], 1)
                image_t = np.tile(image_t, (1, 1, 3))
                image_t = np.transpose(image_t, (2, 0, 1))
            images.append(image_t)
        return np.asarray(images)

    def normalize_data(self, data):
        data0_2 = data / 128.0
        data_norm = data0_2 - 1.0
        # data_norm = data / 255.0
        # data_norm = data - 1.0
        return data_norm


    def read_1_data_and_convert_RGB(self, dir, filename_list, extension, width, height):
        images = []
        for filename in filename_list:
            pilIn = Image.open(dir + filename[0] + extension).convert('RGB')
            pilResize = pilIn.resize((width, height))
            image = np.asarray(pilResize)
            image_t = np.transpose(image, (2, 0, 1))
            images.append(image_t)
        return np.asarray(images)


    def write_data_to_img(self, dir, np_arrays, extension):

        for num, np_array in enumerate(np_arrays):
            pil_img = Image.fromarray(np_array)
            pil_img.save(dir + 'debug_' + str(num) + extension)


    def convert_to_0_1_class_(self, d):
        d_mod = np.zeros((d.shape[0], d.shape[1], d.shape[2], self.class_num), dtype=np.float32)

        for num, image1 in enumerate(d):
            for h, row in enumerate(image1):
                for w, ele in enumerate(row):
                    if int(ele) == 255:#border
                    # if int(ele) == 255 or int(ele) == 0:#border and backgrounds
                        # d_mod[num][h][w][20] = 1.0
                        continue
                    # d_mod[num][h][w][int(ele) - 1] = 1.0
                    d_mod[num][h][w][int(ele)] = 1.0

        return d_mod


    def make_data_for_1_epoch(self):
        self.image_filesX_1_epoch = random.sample(self.file_listX, self.image_fileX_num)
        self.image_filesY_1_epoch = random.sample(self.file_listY, self.image_fileY_num)


    def get_data_for_1_batch(self, i, batchsize, train_FLAG=True):
        # print("len(self.train_files_1_epoch)", len(self.train_files_1_epoch))
        # if train_FLAG:
        data_batchX = self.image_filesX_1_epoch[i:i + batchsize]
        data_batchY = self.image_filesY_1_epoch[i:i + batchsize]
        # else:
        #     print("okasii")
            # data_batch = self.list_val_files[i:i + batchsize]
        imagesX = self.read_1_data(self.dirX, data_batchX, self.img_width, self.img_height)
        imagesY = self.read_1_data(self.dirY, data_batchY, self.img_width, self.img_height)

        imagesX_n = self.normalize_data(imagesX)
        imagesY_n = self.normalize_data(imagesY)

        # labels_0_1 = self.convert_to_0_1_class_(labels)
        return imagesX_n, imagesY_n

    def make_img_from_label(self, labels, epoch):#labels=(first_number, last_number + 1)
        labels_train = self.train_files_1_epoch[labels[0]:labels[1]]
        labels_val = self.list_val_files[labels[0]:labels[1]]
        labels_train_val = labels_train + labels_val
        labels_img_np = self.read_1_data_and_convert_RGB(self.SegmentationClass_dir, labels_train_val, '.png', self.img_width, self.img_height)
        self.write_data_to_img('debug/label_' + str(epoch) + '_',  labels_img_np, '.png')

    def make_img_from_prob(self, probs, epoch):#probs=(data, height, width)..0-20 value
        # print("probs[0]", probs[0])
        print("probs[0].shape", probs[0].shape)
        probs_RGB = util.convert_indexColor_to_RGB(probs)



        # labels_img_np = self.read_1_data_and_convert_RGB(self.SegmentationClass_dir, probs_RGB, '.jpg', self.img_width, self.img_height)
        self.write_data_to_img('debug/prob_' + str(epoch), probs_RGB, '.jpg')


if __name__ == '__main__':
    #debug
    base_dir = '/media/webfarmer/HDCZ-UT/dataset/food101/food-101/images/'
    image_dirX = 'takoyaki/'
    image_dirY = 'macarons/'
    img_width = 128
    img_height = 128


    Make_datasets_Food101 = Make_datasets_Food101(base_dir, img_width, img_height, image_dirX, image_dirY)

    Make_datasets_Food101.make_data_for_1_epoch()
    imagesX, imagesY = Make_datasets_Food101.get_data_for_1_batch(10, 5)
    print("images.shape", imagesX.shape)
    print("labels.shape", imagesY.shape)
    print("labels.dtype", imagesX.dtype)
    print("images[4]", imagesX[4])
    print("labels[4]", imagesY[4])

    image_debug = Image.fromarray(imagesY[2])
    image_debug.show()