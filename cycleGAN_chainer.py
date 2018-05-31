import numpy as np
import os
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
# from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
# from chainer.training import extensions
# from PIL import Image
import utility as Utility
from make_datasets import Make_datasets_Food101
import argparse


def parser():
    parser = argparse.ArgumentParser(description='analyse oyster images')
    parser.add_argument('--batchsize', '-b', type=int, default=1, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='log01', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='epoch')
    parser.add_argument('--test_class_num', '-tc', type=int, default=-1, help='test class number')
    # parser.add_argument('--model_name', '-mn', type=str, default='VGGNet', help='model name')
    parser.add_argument('--tensorboard_log', '-tb', type=str, default='data_18052901', help='directory name of tensorboard')
    parser.add_argument('--base_dir', '-bd', type=str, default='/media/webfarmer/HDCZ-UT/dataset/food101/food-101/images/',
                        help='base directory name of data-sets')
    parser.add_argument('--img_dirX', '-idX', type=str, default='takoyaki/', help='directory name of image X')
    parser.add_argument('--img_dirY', '-idY', type=str, default='macarons/', help='directory name of image Y')


    return parser.parse_args()
args = parser()


#global variants
BATCH_SIZE = args.batchsize
data_size = 6000
noise_num = 100
class_num = 10
n_epoch = args.epoch
WEIGHT_DECAY = 0.0005
BASE_CHANNEL = 32
IMG_SIZE = 128
BASE_DIR = args.base_dir
DIS_LAST_IMG_SIZE = IMG_SIZE // (2**4)
CO_LAMBDA = 10.0
OUT_PUT_IMG_NUM = 6

keep_prob_rate = 0.5

# mnist_file_name = ["mnist_train_img.npy", "mnist_train_label.npy", "mnist_test_img.npy", "mnist_test_label.npy"]
seed = 1234
np.random.seed(seed=seed)

out_image_dir = './out_images_cycleGAN' #output image file
out_model_dir = './out_models_cycleGAN' #output model file

try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
    os.mkdir('./out_images_Debug') #for debug
except:
    print("mkdir error")
    pass

make_data = Make_datasets_Food101(BASE_DIR, IMG_SIZE, IMG_SIZE, image_dirX=args.img_dirX, image_dirY=args.img_dirY)


#generator X------------------------------------------------------------------
class GeneratorX2Y(chainer.Chain):
    def __init__(self):
        super(GeneratorX2Y, self).__init__(
            # First Convolution
            convInit=L.Convolution2D(3, BASE_CHANNEL, ksize=7, stride=1, pad=3),  # 128x128 to 128x128
            # Down 1
            downConv1=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1),
            # Down 2
            downConv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=3, stride=2, pad=1),
            # Residual Block1
            res1Conv1 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res1Conv2 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block2
            res2Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res2Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block3
            res3Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res3Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block4
            res4Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res4Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block5
            res5Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res5Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block6
            res6Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res6Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Up 1
            upConv1=L.Deconvolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, outsize=(64, 64)),
            # Up 2
            upConv2=L.Deconvolution2D(BASE_CHANNEL * 2, BASE_CHANNEL, ksize=3, stride=2, pad=1, outsize=(128, 128)),
            # Last Convolution
            convLast=L.Convolution2D(BASE_CHANNEL, 3, ksize=7, stride=1, pad=3),  # 128x128 to 128x128

            #batch normalization
            bnCI=L.BatchNormalization(BASE_CHANNEL),
            bnD1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnD2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnU1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnU2=L.BatchNormalization(BASE_CHANNEL),
            bnCL=L.BatchNormalization(3),
        )

    def __call__(self, x, train=True):
        #First Convolution
        h = self.convInit(x)
        h = self.bnCI(h)
        h = F.relu(h)
        #Down 1
        h = self.downConv1(h)
        h = self.bnD1(h)
        h = F.relu(h)
        #Down 2
        h = self.downConv2(h)
        h = self.bnD2(h)
        hd2 = F.relu(h)
        #Residual Block 1
        r1 = self.res1Conv1(hd2)
        r1 = self.bnR1C1(r1)
        r1 = F.relu(r1)
        r1 = self.res1Conv2(r1)
        r1 = self.bnR1C2(r1) + hd2
        r1 = F.relu(r1)
        # Residual Block 2
        r2 = self.res2Conv1(r1)
        r2 = self.bnR2C1(r2)
        r2 = F.relu(r2)
        r2 = self.res2Conv2(r2)
        r2 = self.bnR2C2(r2) + r1
        r2 = F.relu(r2)
        # Residual Block 3
        r3 = self.res3Conv1(r2)
        r3 = self.bnR3C1(r3)
        r3 = F.relu(r3)
        r3 = self.res3Conv2(r3)
        r3 = self.bnR3C2(r3) + r2
        r3 = F.relu(r3)
        # Residual Block 4
        r4 = self.res4Conv1(r3)
        r4 = self.bnR4C1(r4)
        r4 = F.relu(r4)
        r4 = self.res4Conv2(r4)
        r4 = self.bnR4C2(r4) + r3
        r4 = F.relu(r4)
        # Residual Block 5
        r5 = self.res5Conv1(r4)
        r5 = self.bnR5C1(r5)
        r5 = F.relu(r5)
        r5 = self.res5Conv2(r5)
        r5 = self.bnR5C2(r5) + r4
        r5 = F.relu(r5)
        # Residual Block 6
        r6 = self.res6Conv1(r5)
        r6 = self.bnR6C1(r6)
        r6 = F.relu(r6)
        r6 = self.res6Conv2(r6)
        r6 = self.bnR6C2(r6) + r5
        r6 = F.relu(r6)
        # Up 1
        h = self.upConv1(r6)
        h = self.bnU1(h)
        h = F.relu(h)
        # Up 2
        h = self.upConv2(h)
        h = self.bnU2(h)
        h = F.relu(h)
        # Last Convolution
        h = self.convLast(h)
        h = self.bnCL(h)
        h = F.relu(h)

        return h


#generator Y------------------------------------------------------------------
class GeneratorY2X(chainer.Chain):
    def __init__(self):
        super(GeneratorY2X, self).__init__(
            # First Convolution
            convInit=L.Convolution2D(3, BASE_CHANNEL, ksize=7, stride=1, pad=3),  # 128x128 to 128x128
            # Down 1
            downConv1=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1),
            # Down 2
            downConv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=3, stride=2, pad=1),
            # Residual Block1
            res1Conv1 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res1Conv2 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block2
            res2Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res2Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block3
            res3Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res3Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block4
            res4Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res4Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block5
            res5Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res5Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Residual Block6
            res6Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            res6Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1),
            # Up 1
            upConv1=L.Deconvolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, outsize=(64, 64)),
            # Up 2
            upConv2=L.Deconvolution2D(BASE_CHANNEL * 2, BASE_CHANNEL, ksize=3, stride=2, pad=1, outsize=(128, 128)),
            # Last Convolution
            convLast=L.Convolution2D(BASE_CHANNEL, 3, ksize=7, stride=1, pad=3),  # 128x128 to 128x128

            #batch normalization
            bnCI=L.BatchNormalization(BASE_CHANNEL),
            bnD1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnD2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnU1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnU2=L.BatchNormalization(BASE_CHANNEL),
            bnCL=L.BatchNormalization(3),
        )

    def __call__(self, x, train=True):
        #First Convolution
        h = self.convInit(x)
        h = self.bnCI(h)
        h = F.relu(h)
        # print("h.data.shape 1", h.data.shape)
        #Down 1
        h = self.downConv1(h)
        h = self.bnD1(h)
        h = F.relu(h)
        # print("h.data.shape 2", h.data.shape)
        #Down 2
        h = self.downConv2(h)
        h = self.bnD2(h)
        hd2 = F.relu(h)
        # print("h.data.shape3 ", h.data.shape)
        #Residual Block 1
        r1 = self.res1Conv1(hd2)
        r1 = self.bnR1C1(r1)
        r1 = F.relu(r1)
        r1 = self.res1Conv2(r1)
        r1 = self.bnR1C2(r1) + hd2
        r1 = F.relu(r1)
        # print("h.data.shape 4", h.data.shape)
        # Residual Block 2
        r2 = self.res2Conv1(r1)
        r2 = self.bnR2C1(r2)
        r2 = F.relu(r2)
        r2 = self.res2Conv2(r2)
        r2 = self.bnR2C2(r2) + r1
        r2 = F.relu(r2)
        # print("h.data.shape 5", h.data.shape)
        # Residual Block 3
        r3 = self.res3Conv1(r2)
        r3 = self.bnR3C1(r3)
        r3 = F.relu(r3)
        r3 = self.res3Conv2(r3)
        r3 = self.bnR3C2(r3) + r2
        r3 = F.relu(r3)
        # print("h.data.shape 6", h.data.shape)
        # Residual Block 4
        r4 = self.res4Conv1(r3)
        r4 = self.bnR4C1(r4)
        r4 = F.relu(r4)
        r4 = self.res4Conv2(r4)
        r4 = self.bnR4C2(r4) + r3
        r4 = F.relu(r4)
        # print("h.data.shape 7", h.data.shape)
        # Residual Block 5
        r5 = self.res5Conv1(r4)
        r5 = self.bnR5C1(r5)
        r5 = F.relu(r5)
        r5 = self.res5Conv2(r5)
        r5 = self.bnR5C2(r5) + r4
        r5 = F.relu(r5)
        # print("h.data.shape 8", h.data.shape)
        # Residual Block 6
        r6 = self.res6Conv1(r5)
        r6 = self.bnR6C1(r6)
        r6 = F.relu(r6)
        r6 = self.res6Conv2(r6)
        r6 = self.bnR6C2(r6) + r5
        r6 = F.relu(r6)
        # print("h.data.shape 9", h.data.shape)
        # Up 1
        h = self.upConv1(r6)
        h = self.bnU1(h)
        h = F.relu(h)
        # print("h.data.shape 10", h.data.shape)
        # Up 2
        h = self.upConv2(h)
        h = self.bnU2(h)
        h = F.relu(h)
        # print("h.data.shape 11", h.data.shape)
        # Last Convolution
        h = self.convLast(h)
        h = self.bnCL(h)
        h = F.relu(h)
        # print("h.data.shape 12", h.data.shape)
        return h


#discriminator X-----------------------------------------------------------------
class DiscriminatorX(chainer.Chain):
    def __init__(self):
        super(DiscriminatorX, self).__init__(
            conv1=L.Convolution2D(3, BASE_CHANNEL * 2, ksize=4, stride=2, pad=1),
            conv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=4, stride=2, pad=1),
            conv3=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 8, ksize=4, stride=2, pad=1),
            conv4=L.Convolution2D(BASE_CHANNEL * 8, BASE_CHANNEL * 16, ksize=4, stride=2, pad=1),
            conv5=L.Convolution2D(BASE_CHANNEL * 16, 1, ksize=DIS_LAST_IMG_SIZE, stride=2, pad=0),

            # batch normalization
            bnC1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnC2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnC3=L.BatchNormalization(BASE_CHANNEL * 8),
            bnC4=L.BatchNormalization(BASE_CHANNEL * 16),
        )

    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = self.bnC1(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv2(h)
        h = self.bnC2(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv3(h)
        h = self.bnC3(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv4(h)
        h = self.bnC4(h)
        h = F.leaky_relu(h, slope=0.2)
        # print("h.data.shape", h.data.shape)
        h = self.conv5(h)
        h = F.reshape(h, (-1, 1))
        out = F.sigmoid(h)
        return out


#discriminator X-----------------------------------------------------------------
class DiscriminatorY(chainer.Chain):
    def __init__(self):
        super(DiscriminatorY, self).__init__(
            conv1=L.Convolution2D(3, BASE_CHANNEL * 2, ksize=4, stride=2, pad=1),
            conv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=4, stride=2, pad=1),
            conv3=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 8, ksize=4, stride=2, pad=1),
            conv4=L.Convolution2D(BASE_CHANNEL * 8, BASE_CHANNEL * 16, ksize=4, stride=2, pad=1),
            conv5=L.Convolution2D(BASE_CHANNEL * 16, 1, ksize=DIS_LAST_IMG_SIZE, stride=2, pad=0),

            # batch normalization
            bnC1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnC2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnC3=L.BatchNormalization(BASE_CHANNEL * 8),
            bnC4=L.BatchNormalization(BASE_CHANNEL * 16),
        )

    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = self.bnC1(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv2(h)
        h = self.bnC2(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv3(h)
        h = self.bnC3(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv4(h)
        h = self.bnC4(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv5(h)
        h = F.reshape(h, (-1, 1))
        out = F.sigmoid(h)
        return out


genX2Y = GeneratorX2Y()
genY2X = GeneratorY2X()
disX = DiscriminatorX()
disY = DiscriminatorY()

genX2Y.to_gpu()
genY2X.to_gpu()
disX.to_gpu()
disY.to_gpu()

optimizer_genX2Y = optimizers.Adam(alpha=0.0003, beta1=0.5)
optimizer_disX = optimizers.Adam(alpha=0.0003, beta1=0.5)
optimizer_genY2X = optimizers.Adam(alpha=0.0003, beta1=0.5)
optimizer_disY = optimizers.Adam(alpha=0.0003, beta1=0.5)

optimizer_genX2Y.setup(genX2Y)
optimizer_disX.setup(disX)
optimizer_genY2X.setup(genY2X)
optimizer_disY.setup(disY)

optimizer_genX2Y.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
optimizer_disX.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
optimizer_genY2X.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
optimizer_disY.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))


#training loop
for epoch in range(0, n_epoch):
    sum_loss_gen_total = np.float32(0)
    sum_loss_gen_X = np.float32(0)
    sum_loss_gen_Y = np.float32(0)
    sum_loss_dis_total = np.float(0)
    sum_loss_dis_X = np.float32(0)
    sum_loss_dis_Y = np.float32(0)
    sum_loss_cycle_X2Y = np.float32(0)
    sum_loss_cycle_Y2X = np.float32(0)

    make_data.make_data_for_1_epoch() #shuffle training data
    len_data = min(make_data.image_fileX_num, make_data.image_fileY_num)

    for i in range(0, len_data, BATCH_SIZE):
        # print("now i =", i)
        imagesX_np, imagesY_np = make_data.get_data_for_1_batch(i, BATCH_SIZE)
        # print("imagesX_np.shape", imagesX_np.shape)

        images_X = Variable(cuda.to_gpu(imagesX_np))
        images_Y = Variable(cuda.to_gpu(imagesY_np))
        # stream around generator
        #
        images_X2Y = genX2Y(images_X)
        images_Y2X = genY2X(images_Y)

        #reverse
        images_X2Y2X = genY2X(images_X2Y)
        images_Y2X2Y = genX2Y(images_Y2X)

        #discriminator
        out_dis_X_real = disX(images_X)
        out_dis_Y_real = disY(images_Y)
        out_dis_X_fake = disX(images_Y2X)
        out_dis_Y_fake = disY(images_X2Y)

        #Cycle Consistency Loss
        loss_cycle_X = F.mean(F.absolute_error(images_X, images_X2Y2X))
        loss_cycle_Y = F.mean(F.absolute_error(images_Y, images_Y2X2Y))
        #Adversarial Loss
        # loss_adv_X_dis = F.mean(- F.log(out_dis_X_real) - F.log(1 - out_dis_X_fake))
        # loss_adv_Y_dis = F.mean(- F.log(out_dis_Y_real) - F.log(1 - out_dis_Y_fake))
        # print("np.mean(out_dis_X_fake.data) ", np.mean(out_dis_X_fake.data))
        # loss_adv_X_gen = F.mean(- F.log(out_dis_X_fake))
        # print("loss_adv_X_gen.data, ", loss_adv_X_gen.data)
        # loss_adv_Y_gen = F.mean(- F.log(out_dis_Y_fake))

        #make target for adversarial loss
        tar_1_np = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        tar_0_np = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        tar_1 = Variable(cuda.to_gpu(tar_1_np))
        tar_0 = Variable(cuda.to_gpu(tar_0_np))

        # Adversarial Loss
        loss_adv_X_dis = F.mean_squared_error(out_dis_X_real, tar_1) + F.mean_squared_error(out_dis_X_fake, tar_0)
        loss_adv_Y_dis = F.mean_squared_error(out_dis_Y_real, tar_1) + F.mean_squared_error(out_dis_Y_fake, tar_0)
        loss_adv_X_gen = F.mean_squared_error(out_dis_X_fake, tar_1)
        loss_adv_Y_gen = F.mean_squared_error(out_dis_Y_fake, tar_1)

        #total Loss
        # print("loss_adv_X_gen.data.shape", loss_adv_X_gen.data.shape)
        # print("loss_adv_Y_gen.data.shape", loss_adv_Y_gen.data.shape)
        # print("loss_cycle_X.data.shape", loss_cycle_X.data.shape)
        # print("loss_cycle_Y.data.shape", loss_cycle_Y.data.shape)
        loss_gen_total = loss_adv_X_gen + loss_adv_Y_gen + CO_LAMBDA * (loss_cycle_X + loss_cycle_Y)
        loss_dis_total = loss_adv_X_dis + loss_adv_Y_dis

        # for print
        sum_loss_gen_total += loss_gen_total.data
        sum_loss_gen_X += loss_adv_X_gen.data
        sum_loss_gen_Y += loss_adv_Y_gen.data
        sum_loss_dis_total += loss_dis_total.data
        sum_loss_dis_X += loss_adv_X_dis.data
        sum_loss_dis_Y += loss_adv_Y_dis.data
        sum_loss_cycle_Y2X += loss_cycle_X.data
        sum_loss_cycle_X2Y += loss_cycle_Y.data
        # print("sum_loss_gen_X", sum_loss_gen_X)
        # # print("sum_loss_gen_Y", sum_loss_gen_Y)
        # # print("sum_loss_dis_X", sum_loss_dis_X)
        # # print("sum_loss_dis_Y", sum_loss_dis_Y)
        # # print("sum_loss_cycle_Y2X", sum_loss_cycle_Y2X)
        # # print("sum_loss_cycle_X2Y", sum_loss_cycle_X2Y)

        # discriminator back prop
        disX.cleargrads()
        disY.cleargrads()
        loss_dis_total.backward()
        optimizer_disX.update()
        optimizer_disY.update()

        # generator back prop
        genX2Y.cleargrads()
        genY2X.cleargrads()
        loss_gen_total.backward()
        optimizer_genX2Y.update()
        optimizer_genY2X.update()



    print("----------------------------------------------------------------------")
    print("epoch =", epoch , ", Total Loss of G =", sum_loss_gen_total / len_data, ", Total Loss of D =", sum_loss_dis_total / len_data)
    print("Discriminator: Loss X =", sum_loss_dis_X / len_data, ", Loss Y =", sum_loss_dis_Y / len_data)
    print("Generator: Loss adv X=", sum_loss_gen_X / len_data, ", Loss adv Y =", sum_loss_gen_Y / len_data,)
    print("Generator: Loss cycle Y2X=", sum_loss_cycle_Y2X / len_data, ", Loss cycle X2Y =", sum_loss_cycle_X2Y / len_data,)

    #outupt generated images
    #img_X, img_X2Y, img_X2Y2X, img_Y, img_Y2X, img_Y2X2Y, out_image_dir, epoch
    img_X = []
    img_X2Y = []
    img_X2Y2X = []
    img_Y = []
    img_Y2X = []
    img_Y2X2Y = []
    for i in range(OUT_PUT_IMG_NUM):
        imagesX_np, imagesY_np = make_data.get_data_for_1_batch(i, 1)
        # print("imagesX_np.shape", imagesX_np.shape)

        img_X.append(imagesX_np[0])
        img_Y.append(imagesY_np[0])

        images_X = Variable(cuda.to_gpu(imagesX_np))
        images_Y = Variable(cuda.to_gpu(imagesY_np))

        # stream around generator
        images_X2Y = genX2Y(images_X)
        images_Y2X = genY2X(images_Y)

        img_X2Y.append(images_X2Y.data[0])
        img_Y2X.append(images_Y2X.data[0])

        # reverse
        images_X2Y2X = genY2X(images_X2Y)
        images_Y2X2Y = genX2Y(images_Y2X)

        img_X2Y2X.append(images_X2Y2X.data[0])
        img_Y2X2Y.append(images_Y2X2Y.data[0])

    img_X_np = np.asarray(img_X).transpose((0, 2, 3, 1))
    img_Y_np = np.asarray(img_Y).transpose((0, 2, 3, 1))
    img_X2Y_np = np.asarray(img_X2Y).transpose((0, 2, 3, 1))
    img_Y2X_np = np.asarray(img_Y2X).transpose((0, 2, 3, 1))
    img_X2Y2X_np = np.asarray(img_X2Y2X).transpose((0, 2, 3, 1))
    img_Y2X2Y_np = np.asarray(img_Y2X2Y).transpose((0, 2, 3, 1))

    Utility.make_output_img(img_X_np, img_X2Y_np, img_X2Y2X_np, img_Y_np, img_Y2X_np, img_Y2X2Y_np, out_image_dir, epoch)
    #    Utility.make_output_img(img_X_np, img_Y_np, img_X2Y_np, img_Y2X_np, img_X2Y2X_np, img_Y2X2Y_np, out_image_dir, epoch)


    '''
    if epoch % 10 == 0:
        sample_num_h = 10
        sample_num = sample_num_h ** 2

        # z_test = np.random.uniform(0, 1, sample_num_h * noise_num).reshape(sample_num_h, 1, noise_num)
        # z_test = np.tile(z_test, (1, sample_num_h, 1))
        z_test = np.random.uniform(0, 1, sample_num_h * noise_num).reshape(1, sample_num_h, noise_num)
        z_test = np.tile(z_test, (sample_num_h, 1, 1))
        z_test = z_test.reshape(-1, sample_num).astype(np.float32)
        label_gen_int = np.arange(10).reshape(10, 1).astype(np.float32)
        label_gen_int = np.tile(label_gen_int, (1, 10)).reshape(sample_num)
        label_gen_test = make_mnist.convert_to_10class_(label_gen_int)
        label_gen_test = Variable(cuda.to_gpu(label_gen_test))
        z_test = Variable(cuda.to_gpu(z_test))
        x_gen_test, y_gen_test = gen(label_gen_test, z_test, train=False)
        x_gen_test_data = x_gen_test.data
        x_gen_test_reshape = x_gen_test_data.reshape(len(x_gen_test_data), 28, 28, 1)
        x_gen_test_reshape = cuda.to_cpu(x_gen_test_reshape)
        Utility.make_output_img(x_gen_test_reshape, sample_num_h, out_image_dir, epoch)

    if epoch % 100 == 0:
        #serializer
        serializers.save_npz(out_model_dir + '/gen_' + str(epoch) + '.model', gen)
        serializers.save_npz(out_model_dir + '/cla_' + str(epoch) + '.model', cla)
        serializers.save_npz(out_model_dir + '/dis_' + str(epoch) + '.model', dis)
    '''
