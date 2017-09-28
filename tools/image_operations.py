# -*- coding: utf-8 -*-
from libtiff import TIFF
from PIL import Image
import numpy as np
import cairosvg
import gc
import os
from skimage.measure import regionprops
from glob import glob


# 读取ｔｉｆｆ文件
def tiff_read(image_path, new_size=None):
    if image_path.endswith('.jpg') or image_path.endswith('.png'):
        image = Image.open(image_path)
        if new_size is not None:
            image = image.resize(new_size)
        return np.array(
            image
        )
    tif = TIFF.open(image_path)
    idx = 0
    images = []
    for image in list(tif.iter_images()):
        images.append(image)
    if len(images) != 1:
        print 'the images of tiff image more than one'
        return None
    return np.array(images[0])


# 读取多个tiff文件
def tiffs_read(images_path, new_size=None):
    images = []
    for image_path in images_path:
        images.append(tiff_read(image_path, new_size))
    return np.array(images)


# 保存tiff文件
def tiff_save(image, path):
    if path.endswith('tiff'):
        tiff_file = TIFF.open(path, mode='w')
        tiff_file.write_image(image, compression=None, write_rgb=True)
        tiff_file.close()
    else:
        img = Image.fromarray(image)
        img.save(path)


# 将ＳＶＧ文件转化为ＰＮＧ图片
def conver_svg_png(svg_path, png_path):
    print svg_path
    fileHandle = open(svg_path)
    svg = fileHandle.read()
    cairosvg.svg2png(bytestring=svg, write_to=png_path)
    print 'successfully export', svg_path


# 将所有的svg文件都转化为png图片
def conver_all_svgs(svgs_path, png_dir_path):
    svg_files = os.listdir(svgs_path)
    for svg_name in svg_files:
        svg_path = os.path.join(svgs_path, svg_name)
        base_name = os.path.basename(svg_path).split('.tiff')[0]
        png_path = os.path.join(png_dir_path, base_name + '.png')
        conver_svg_png(svg_path, png_path)


# 将tiff降低分辨率
def resize_tiff(tiff_path, new_tiff_path, new_size):
    tiff_image = tiff_read(tiff_path)
    image = Image.fromarray(tiff_image)
    image = image.resize(new_size)
    tiff_save(image, new_tiff_path)


# 将单个文件夹的ｔｉｆｆ文件全部ｒｅｓｉｚｅ
def resize_dir(tiff_dir, new_tiff_dir, new_size):
    tiff_files = os.listdir(tiff_dir)
    for tiff_file in tiff_files:
        resize_tiff(
            os.path.join(tiff_dir, tiff_file),
            os.path.join(new_tiff_dir, tiff_file),
            new_size
        )


# 填充图像
def fill_region(png_path):
    image = Image.open(png_path)
    image = np.array(image)
    new_image = image[:, :, 0] + image[:, :, 1] + image[:, :, 2]
    # image.show()
    from scipy import ndimage
    image = ndimage.binary_fill_holes(new_image).astype(np.uint8)
    image = Image.fromarray(image * 255)
    image.show()


# 针对每一个图片提取ｐａｔｃｈ
def extract_patch_single_tiff(tiff_path, png_path, save_path, patch_size=64, stride_size=8, image_save_dir=None):
    tiff_image = tiff_read(tiff_path)
    patch_record = open(save_path, 'a')
    shape = list(np.shape(tiff_image))
    w = shape[0]
    h = shape[1]
    if not os.path.exists(png_path):
        png_exists = False
        png_mask = np.zeros([w, h], dtype=np.uint8)

    else:
        png_exists = True
        png_mask = Image.open(png_path)
        png_mask = np.array(png_mask)
    labels = []
    basename = os.path.basename(tiff_path)
    basename = basename[:basename.find('.tiff')]
    count = 0
    # print np.shape(tiff_image)
    for x in range(patch_size/2, w-patch_size/2, stride_size):
        for y in range(patch_size/2, h-patch_size/2, stride_size):
            if png_exists and png_mask[x, y] == 255:
                labels.append(1)
                cur_label = '1'
            else:
                labels.append(0)
                cur_label = '0'
            patch_mask = png_mask[x-patch_size/2:x+patch_size/2, y-patch_size/2: y+patch_size/2]
            score = (1.0 * np.sum(patch_mask != 0)) / (patch_size * patch_size)
            if score == 0.0 and not (os.path.basename(tiff_path).startswith('Normal') or os.path.basename(tiff_path).startswith('normal')):
                continue
            x_d = x - patch_size / 2
            x_t = x + patch_size / 2
            y_d = y - patch_size / 2
            y_t = y + patch_size / 2
            count += 1
            position_str = str(x_d) + ' ' + str(x_t) + ' ' + str(y_d) + ' ' + str(y_t)
            patch_record.write(
                basename + ' ' + str(count) + ' ' + cur_label + ' ' + str(score) + ' ' + position_str + '\n'
            )
            if image_save_dir is not None:
                patch_save_path = os.path.join(
                    image_save_dir, basename + '_' + str(count) + '_' + cur_label + '_' + str(score) + '.tiff'
                )
                cur_path = tiff_image[x-patch_size/2:x+patch_size/2, y-patch_size/2: y+patch_size/2, :]
                tiff_save(cur_path, patch_save_path)
    # np.save(save_path[0], patches)
    # np.save(save_path[1], labels)
    patch_record.close()
    print tiff_path, ' extract patches finshed, saved in ', save_path, ' count is ', count
    return


def extract_patch_single_dir(dir_path, patch_record_file, png_dir, patch_image_save_dir):
    tiff_names = os.listdir(dir_path)
    for tiff_name in tiff_names:
        tiff_path = os.path.join(dir_path, tiff_name)
        base_name = tiff_name[:tiff_name.find('.tiff')]
        png_path = os.path.join(
            png_dir,
            base_name + '.png'
        )

        extract_patch_single_tiff(
            tiff_path,
            png_path,
            patch_record_file,
            patch_size=256,
            stride_size=64,
            image_save_dir=patch_image_save_dir
        )


def extract_patches_multi_dir(dir_paths, save_paths, png_dir, patches_save_image_paths=[None, None, None, None]):
    for index, dir_path in enumerate(dir_paths):
        extract_patch_single_dir(
            dir_path,
            save_paths[index],
            png_dir,
            patches_save_image_paths[index]
        )


# 处理多个文件夹
def resize_multi_dir(dirs, new_size):
    resize_dir(
        '/home/give/Documents/dataset/BOT_Game/data_NY/train/negative',
        dirs[0],
        new_size
    )
    resize_dir(
        '/home/give/Documents/dataset/BOT_Game/data_NY/train/positive',
        dirs[1],
        new_size
    )
    resize_dir(
        '/home/give/Documents/dataset/BOT_Game/data_NY/original/negative',
        dirs[2],
        new_size
    )
    resize_dir(
        '/home/give/Documents/dataset/BOT_Game/data_NY/original/positive',
        dirs[3],
        new_size
    )


# 提取绝对是癌症的区域作为patch
def extract_absolute_positive(tiff_path, png_path, save_path, stride_size=64, patch_size=256):
    tiff_base_name = os.path.basename(tiff_path).split('.tiff')[0]
    tiff_image = tiff_read(tiff_path)
    shape = list(np.shape(tiff_image))
    w = shape[0]
    h = shape[1]
    if png_path is not None:
        mask_image = np.array(Image.open(png_path))
    else:
        # 肯定不是癌症的话，肯定都可以取
        mask_image = np.ones([w, h])
    count = 0
    for x in range(0, w-patch_size, stride_size):
        for y in range(0, h-patch_size, stride_size):
            cur_mask = mask_image[x:x+patch_size, y:y+patch_size]
            if 0 in cur_mask:
                # 只选取绝对的癌症区域
                continue
            #　print x, y
            cur_patch = tiff_image[x:x+patch_size, y:y+patch_size, :]
            cur_patch_path = os.path.join(
                save_path, tiff_base_name + '_' + str(count) + '.tiff'
            )
            tiff_save(cur_patch, cur_patch_path)
            count += 1
    print tiff_path, 'finished extract_absolute_positive. patch number is ', count


# 提取一个文件夹中的所有的patch
def extract_absolution_positive_dirs(dirs, pngs, save_path, stride_size=64, patch_size=256):
    tiff_names = os.listdir(dirs)
    for tiff_name in tiff_names:
        tiff_base_name = tiff_name.split('.tiff')[0]
        if pngs is not None:
            png_path = os.path.join(pngs, tiff_base_name+'.png')
        else:
            png_path = None
        extract_absolute_positive(
            os.path.join(dirs, tiff_name),
            png_path,
            save_path,
            stride_size=stride_size,
            patch_size=patch_size
        )


# 计算训练集合的平均像素值密度
def calu_average_train_set(train_dir, new_size=None):
    positive_dir = os.path.join(train_dir, 'positive')
    positive_files = os.listdir(positive_dir)
    images = None
    count = 0
    for positive_file in positive_files:
        tiff_path = os.path.join(positive_dir, positive_file)
        cur_image = Image.fromarray(
            tiff_read(tiff_path)
        )
        if new_size is not None:
            cur_image = cur_image.resize(new_size)
        cur_image = np.asarray(cur_image, np.float32)
        # print tiff_path
        if images is None:
            images = cur_image
        else:
            images += cur_image
        count += 1
    print 'have finished positive'
    negative_dir = os.path.join(train_dir, 'negative')
    negative_files = os.listdir(negative_dir)
    for negative_file in negative_files:
        tiff_path = os.path.join(negative_dir, negative_file)
        cur_image = Image.fromarray(
            tiff_read(tiff_path)
        )
        if new_size is not None:
            cur_image = cur_image.resize(new_size)
        cur_image = np.asarray(cur_image, np.float32)
        # print tiff_path
        if images is None:
            images = cur_image
        else:
            images += cur_image
        count += 1
    print np.shape(images)
    image_avg = images / count
    del images
    gc.collect()
    return image_avg



# 将数据按照指定的方式排序
def changed_shape(images, shape):
    new_image = np.zeros(
        shape=shape
    )
    batch_size = shape[0]
    for z in range(batch_size):
        for phase in range(shape[-1]):
            if shape[-1] == 1:
                new_image[z, :, :, phase] = images[z]
            else:
                new_image[z, :, :, phase] = images[z, phase]
    del images
    gc.collect()
    return new_image


# 将一个图像转化成指定的模型
def convert_image_type(image_path, type_name):
    base_name = os.path.basename(image_path)
    if base_name.find('.tiff') != -1:
        file_name = base_name.split('.tiff')[0]
        cur_type = 'tiff'
    else:
        file_name = base_name.split('.')[0]
        cur_type = base_name.split('.')[1]
    if cur_type == type_name:
        return
    print cur_type
    image = tiff_read(image_path)
    tiff_save(
        image,
        os.path.join(os.path.dirname(image_path), file_name + '.' + type_name)
    )


# 得到该目录下面所有文件的路径
def load_image_names(path_name):
    image_names = os.listdir(path_name)
    return [
        os.path.join(path_name, image_name) for image_name in image_names
    ]


# 将文件夹下面的所有文件都转化成指定类型
def convert_images_type(image_dir, type_name):
    image_names = load_image_names(image_dir)
    for image_name in image_names:
        convert_image_type(image_name, type_name)


# 查看文件下有没有同名文件
def has_same_file(path1, path2):
    names1 = os.listdir(path1)
    names2 = os.listdir(path2)
    for name in names1:
        if name in names2:
            return True
    return False

# 保存image
def save_image(image, path):
    img = Image.fromarray(image)
    img.save(path)

# 从图像中提取病灶patch, 并且返回
'''
    :param tiff_path 图像的路径
    :param mask_dir 如果是train and mask_dir 的话那就需要maskdir 否则是None
    :param patch_size patch的大小
    :param stride 步长的大小
    :param occupy_rate positive区域占据的比例
'''
def extract_patchs_return(tiff_path, mask_dir, patch_size, stride, occupy_rate):
    tiff_image = tiff_read(tiff_path)
    basename = os.path.basename(tiff_path).split('.tiff')[0]
    if mask_dir is not None:
        mask_path = os.path.join(mask_dir, basename+'.png')
        mask_image = tiff_read(mask_path)
    shape = list(np.shape(tiff_image))
    count = 0
    patches = []
    for i in range(patch_size/2, shape[0]-patch_size/2, stride):
        for j in range(patch_size / 2, shape[1] - patch_size / 2, stride):
            if mask_dir is not None:
                if mask_image[i, j] == 0:
                    continue
                cur_patch = tiff_image[i-patch_size/2:i+patch_size/2, j-patch_size/2:j+patch_size/2]
                cur_mask = mask_image[i-patch_size/2:i+patch_size/2, j-patch_size/2:j+patch_size/2]
                cur_rate = ((1.0 * np.sum(cur_mask != 0)) / (1.0 * patch_size * patch_size))
                if cur_rate >= occupy_rate:
                    count += 1
                    patches.append(cur_patch)
            else:
                cur_patch = tiff_image[i - patch_size / 2:i + patch_size / 2, j - patch_size / 2:j + patch_size / 2]
                patches.append(cur_patch)
    return patches

# 从图像中提取病灶patch, 并且保存到相应的目录
def extract_patchs_save(tiff_path, mask_dir, save_dir, patch_size, stride, occupy_rate):
    tiff_image = tiff_read(tiff_path)
    if os.path.basename(tiff_path).endswith('.tiff'):
        basename = os.path.basename(tiff_path).split('.tiff')[0]
    else:
        basename = os.path.basename(tiff_path).split('.png')[0]
    if mask_dir is not None:
        mask_path = os.path.join(mask_dir, basename+'.png')
        mask_image = tiff_read(mask_path)
    shape = list(np.shape(tiff_image))
    count = 0
    for i in range(patch_size/2, shape[0]-patch_size/2, stride):
        for j in range(patch_size / 2, shape[1] - patch_size / 2, stride):
            if mask_dir is not None:
                if mask_image[i, j] == 0:
                    continue
                cur_patch = tiff_image[i-patch_size/2:i+patch_size/2, j-patch_size/2:j+patch_size/2]
                cur_mask = mask_image[i-patch_size/2:i+patch_size/2, j-patch_size/2:j+patch_size/2]
                cur_rate = ((1.0 * np.sum(cur_mask != 0)) / (1.0 * patch_size * patch_size))
                if cur_rate >= occupy_rate:
                    count += 1
                    cur_rate = '%.3f' % cur_rate
                    save_path = os.path.join(save_dir, basename+'_' + str(i) + '_' + str(j) + '_' + cur_rate +'.png')
                    save_image(cur_patch, save_path)
            else:
                cur_patch = tiff_image[i - patch_size / 2:i + patch_size / 2, j - patch_size / 2:j + patch_size / 2]
                save_path = os.path.join(save_dir, basename + '_' + str(i) + '_' + str(j) + '.png')
                save_image(cur_patch, save_path)
    return count

# 遍历整个文件夹下面所有的positive样本，切割出满足条件的patch
def extract_patches_one_folder(tiff_dir, mask_dir, save_dir, patch_size, stride, occupy_rate):
    names = os.listdir(tiff_dir)
    count_zero_pathes = []
    tiff_pathes = [os.path.join(tiff_dir, name) for name in names]
    for tiff_path in tiff_pathes:
        count = extract_patchs_save(tiff_path, mask_dir, save_dir, patch_size, stride, occupy_rate)
        if count == 0:
            count_zero_pathes.append(tiff_path)
        print 'extract finish, ', tiff_path, ' the number of patches: ', count
    print 'zero pathes is ', count_zero_pathes

# 遍历整个文件夹下面的所有negative样本，指定需要提取的总数
'''
    :param tiff_dir tiff文件所在的文件夹
    :param save_dir 保存的路径
    :param all_sum 指定的个数
'''
def extract_patches_one_folder_num(tiff_dir, save_dir, all_num, patch_size, process_num=8):
    def single_process(pathes, mask_dir, stride, patch_size, save_dir, rate):
        for tiff_path in pathes:
            print 'extract finish, ', tiff_path, 'the numebr of patches is ', per_file_num
            extract_patchs_save(
                tiff_path,
                mask_dir=mask_dir,
                stride=stride,
                patch_size=patch_size,
                save_dir=save_dir,
                occupy_rate=rate,
            )
    names = os.listdir(tiff_dir)
    name_num = len(names)
    per_file_num = all_num/name_num
    import math
    stride = int(2048/math.sqrt(per_file_num))
    tiff_pathes = [os.path.join(tiff_dir, name) for name in names]
    start = 0
    per_process_num = len(tiff_pathes) / process_num + 1
    from multiprocessing import Process
    for i in range(process_num):
        end = start + per_process_num
        if end > len(tiff_pathes):
            end = len(tiff_pathes)
        cur_pathes = tiff_pathes[start: end]
        start = end
        process = Process(
            target=single_process,
            args=[
                cur_pathes,
                None,
                stride,
                patch_size,
                save_dir,
                None
            ]
        )
        process.start()
'''
    删除癌症patch中包含的可能是正常细胞的区域
    :param patch_dir 存放patch 的文件夹
    :param rate 白色区域存在的比例 如果大于该比例，则认为可以删除该patch
'''
def delete_positive_pathc(patch_dir, rate, process_num=8):
    from multiprocessing import Process
    def single_process(patches):
        for patch_path in patches:
            tiff_image = tiff_read(patch_path)
            shape = list(np.shape(tiff_image))
            equal_map = tiff_image >= [240, 240, 240]
            equal_sum = np.sum(equal_map, axis=2)
            equal_map = equal_sum == 3
            equal_sum = np.sum(equal_map)
            if ((1.0 * equal_sum) / (1.0 * shape[0] * shape[1])) >= rate:
                os.remove(patch_path)
                print 'delete: ', patch_path
            # else:
            #     print 'remain: ', patch_path
    patch_names = os.listdir(patch_dir)
    patch_pathes = [os.path.join(patch_dir, patch_name) for patch_name in patch_names]
    per_process_num = (len(patch_pathes) / process_num) + 1
    start = 0
    for i in range(process_num):
        end = start + per_process_num
        if end > len(patch_pathes):
            end = len(patch_pathes)
        processed_patch = patch_pathes[start: end]
        process = Process(target=single_process, args=[processed_patch, ])
        process.start()
        start = end
'''
    读取一个文件，这个图像文件可能是tiff格式也有可能是jpg或者是png
'''
def read_image(image_path):
    if not image_path.endswith('.tiff'):
        image = Image.open(image_path)
        return image
    else:
        return Image.fromarray(tiff_read(image_path))

'''
    针对图像我们去做直方图均衡化
'''
def histogram_equalization(img):
    import cv2
    eq = cv2.equalizeHist(img)
    return eq


# 将一副胃癌图像转化为hsv，然后再在v channel做直方图均衡化，最后在转化为RGB格式
def rgb_histogram_equlization(rgb_image_path, save_image_dir):
    image = read_image(rgb_image_path)
    image_hsv = image.convert('HSV')
    image_hsv_arr = np.array(image_hsv)
    v_channel = image_hsv_arr[: ,:, 2]
    v_channel_eq = histogram_equalization(v_channel)
    image_hsv_arr[:, :, 2] = v_channel_eq
    im1 = Image.fromarray(image_hsv_arr[:, :, 0])
    im2 = Image.fromarray(image_hsv_arr[:, :, 1])
    im3 = Image.fromarray(image_hsv_arr[:, :, 2])
    image_hsv_eq = Image.merge('HSV', (im1, im2, im3))
    image_eq = image_hsv_eq.convert('RGB')
    image_eq.save(os.path.join(save_image_dir, os.path.basename(rgb_image_path).split('.tiff')[0] + '.png'))
    print 'save eq image, ', os.path.join(save_image_dir, os.path.basename(rgb_image_path))


'''
    将一个文件夹下面的所有图像均衡化, 保存为PNG格式
'''
def rgb_histogram_equlization_folder(rgb_image_dir, save_image_dir, process_num=8):
    def single_process(pathes):
        for path in pathes:
            rgb_histogram_equlization(path, save_image_dir)
    names = os.listdir(rgb_image_dir)
    pathes = [os.path.join(rgb_image_dir, name) for name in names]
    start = 0
    per_process_num = (len(pathes) / process_num) + 1
    from multiprocessing import Process
    for i in range(process_num):
        end = start + per_process_num
        if end > len(pathes):
            end = len(pathes)
        cur_path = pathes[start:end]
        start = end
        process = Process(target=single_process, args=[cur_path, ])
        process.start()
'''
    将一个文件夹下面的所有tiff文件转化为PNG格式
'''
def convert_format(tiff_dir, save_dir, process_num=10):
    def convert_one_file(tiff_path, save_dir):
        tiff_image = tiff_read(tiff_path)
        img = Image.fromarray(tiff_image)
        basename = os.path.basename(tiff_path).split('.tiff')[0] + '.png'
        img.save(os.path.join(save_dir, basename))
        print 'convert finish', tiff_path
    def single_process(pathes):
        for path in pathes:
            convert_one_file(path, save_dir)
    names = os.listdir(tiff_dir)
    pathes = [os.path.join(tiff_dir, name) for name in names]
    start = 0
    per_process_num = (len(pathes) / process_num) + 1
    from multiprocessing import Process
    for i in range(process_num):
        end = start + per_process_num
        if end > len(pathes):
            end = len(pathes)
        cur_path = pathes[start:end]
        start = end
        process = Process(target=single_process, args=[cur_path, ])
        process.start()

'''
    将单个癌症区域mask文件拆分，因为原来可能存在在一个mask文件中标注了多个不连通的癌症区域的情况
    :param mask_path 原先mask文件的路径
    :param save_dir 存放的路径
    :return 返回连通区域的个数
'''
def split_connected_region(mask_path, save_dir):
    mask_image = read_image(mask_path)
    mask_image = mask_image.convert('L')
    from skimage.measure import label
    label_img = label(np.array(mask_image))
    label_arr = np.asarray(label_img, np.uint8)
    # (Image.fromarray(np.asarray(label_img, np.uint8) * 20)).show()
    regions = regionprops(np.array(label_img))
    basename = os.path.basename(mask_path).split('.png')[0]
    count = 0
    for index in range(1, len(regions)):
        new_mask_img = np.zeros(
            list(np.shape(mask_image)),
            np.uint8
        )
        new_mask_img[label_arr == index] = 255
        (Image.fromarray(new_mask_img)).save(
            os.path.join(save_dir, basename + '_' + str(index) + '.png')
        )
        count += 1
    return count
'''
    将癌症区域拆分，原来是将多个癌症区域画在一个label中的，经过此函数的处理将不同区域的（不连通的区域）分别花在不同的文件里面。
    :param label_dir 原来的label文件存放的文件夹
    :param save_dir 新存放的文件夹
'''
def split_connected_regions(label_dir, save_dir, process_num=8):
    def single_process(pathes, save_dir):
        for path in pathes:
            connected_region_count = split_connected_region(path, save_dir)
            print path, ' ', connected_region_count
    label_names = os.listdir(label_dir)
    label_pathes = [os.path.join(label_dir, label_name) for label_name in label_names]
    start = 0
    per_process_num = (len(label_pathes) / process_num) + 1
    from multiprocessing import Process
    for i in range(process_num):
        end = start + per_process_num
        if end > len(label_pathes):
            end = len(label_pathes)
        print 'pid: %d, %d ~ %d' % (i, start, end)
        cur_pathes = label_pathes[start:end]
        start = end
        process = Process(
            target=single_process,
            args=[cur_pathes, save_dir, ]
        )
        process.start()

'''
    根据单个联通区域来提取patch，和上面的不同之处在于，上面的patch的提取所使用的mask是多个连通区域混合在一起的
'''
def extract_patches_single_connected_region(mask_path, image_dirs, save_dirs, patch_size, stride_size, occury_rate):
    mask_image = np.array(read_image(mask_path))
    basename = os.path.basename(mask_path).split('.2048x2048')[0] + '.2048x2048'
    origin_image_path = os.path.join(image_dirs[0], basename+'.tiff')
    position_flag = 0
    if not os.path.exists(origin_image_path):
        origin_image_path = os.path.join(image_dirs[1], basename + '.tiff')
        position_flag = 1
        if not os.path.exists(origin_image_path):
            assert 'not exists'
    origin_image = np.array(read_image(origin_image_path))
    shape = list(np.shape(origin_image))
    count = 0
    for i in range(patch_size/2, shape[0]-patch_size/2, stride_size):
        for j in range(patch_size/2, shape[1]-patch_size/2, stride_size):
            if mask_image[i, j] == 0:
                continue
            cur_patch = origin_image[i - patch_size / 2:i + patch_size / 2, j - patch_size / 2:j + patch_size / 2]
            cur_mask = mask_image[i - patch_size / 2:i + patch_size / 2, j - patch_size / 2:j + patch_size / 2]
            cur_rate = ((1.0 * np.sum(cur_mask != 0)) / (1.0 * patch_size * patch_size))
            if cur_rate >= occury_rate:
                count += 1
                cur_rate = '%.3f' % cur_rate
                save_path = os.path.join(save_dirs[position_flag], basename + '_' + str(i) + '_' + str(j) + '_' + cur_rate + '.png')
                save_image(cur_patch, save_path)
    if count == 0:
        xs, ys = np.where(mask_image != 0)
        xs_min = np.min(xs)
        xs_max = np.max(xs)
        ys_min = np.min(ys)
        ys_max = np.max(ys)
        cur_patch = origin_image[xs_min: xs_max, ys_min: ys_max]
        if xs_max == xs_min or ys_max == ys_min:
            return count
        print mask_path, np.shape(cur_patch)
        save_image(cur_patch, os.path.join(save_dirs[position_flag], basename + '_' + str(2048) + '_' + str(2048) + '.png'))
        count += 1
    return count
'''
     处理的单位是文件夹，根据单个联通区域来提取patch
     :param mask_dir 单连通区域mask图像存放的文件夹
     :param image_dirs 图像可能存在的路径,由于我们将图像拆分成了train 和val所以这里会是数组的形式
     :param save_dirs 存放的路径
     :param patch_size 提取patzh 的大小
     :param stride_size 步长
     :param occury_rate 占比，大于该比率才保存该patch
     :param process_num 多进程处理时候的进程个数
'''
def extract_patches_single_connected_regions(mask_dir, image_dirs, save_dirs, patch_size, stride_size, occury_rate, process_num=8):
    def single_process(pathes):
        for path in pathes:
            patches_num = extract_patches_single_connected_region(path, image_dirs, save_dirs, patch_size, stride_size, occury_rate)
            print path, ' ', patches_num
    mask_names = os.listdir(mask_dir)
    mask_pathes = [os.path.join(mask_dir, mask_name) for mask_name in mask_names]
    start = 0
    per_process_num = (len(mask_pathes) / process_num) + 1
    from multiprocessing import Process
    for i in range(process_num):
        end = start + per_process_num
        if end > len(mask_pathes):
            end = len(mask_pathes)
        print 'pid: %d, %d ~ %d' % (i, start, end)
        cur_pathes = mask_pathes[start:end]
        start = end
        process = Process(
            target=single_process,
            args=[cur_pathes, ]
        )
        process.start()
'''
    根据一个数组显示直方图
'''
def show_histogram(arr):
    arr = list(arr)
    arr_set = list(set(arr))
    arr_dict = {}
    for key in arr_set:
        arr_dict[key] = arr.count(key)
    import matplotlib.pyplot as plt
    keys = list(arr_dict.keys())
    keys.sort()
    plt.bar(keys, [arr_dict[key] for key in keys], width=1.0, align='center', color='red')
    plt.show()
'''
    统计图像的大小，并以灰度分布直方图的形式展现出来
'''
def calu_static_areas(mask_dir):
    mask_names = os.listdir(mask_dir)
    mask_pathes = [os.path.join(mask_dir, mask_name) for mask_name in mask_names]
    areas = []
    records = {}
    for mask_path in mask_pathes:
        # print mask_path
        mask_image = np.array(read_image(mask_path))
        mask_sum = np.sum(mask_image != 0)
        if mask_sum <= 10:
            continue
        areas.append(
            mask_sum
        )
        records[mask_path] = mask_sum
    areas.sort()
    areas = areas[:int(0.75*len(areas))]    # 只选取前百分之75
    opened_file = open('./static_areas.txt', 'w')
    lines = [key + ' ' + str(records[key]) + '\n' for key in records.keys()]
    opened_file.writelines(lines)
    print 'average_areas: ', np.sum(areas) / (len(areas))
    show_histogram(areas)

'''
    对我们mask文件进行分类，分成连类，一类是完全可以容纳在256*256框内的，一类是不能完全容纳在256*256框内的
    实现方法：找出ROI的长和宽，if max(w, h) > 256 则是第二类，否则是第一类
'''
def splited_labels_into_categorys(label_dir, category_dir1, category_dir2, process_num=8):
    def single_process(label_pathes, category_dir1, category_dir2):
        for label_path in label_pathes:
            mask_image = read_image(label_path)
            mask_image = np.array(mask_image)
            if np.sum(mask_image == 255) <= 100:
                print 'jump on: ', label_path, ' sum is: ', np.sum(mask_image == 255)
                continue
            xs, ys = np.where(mask_image != 0)
            limit = max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))
            if limit > 256:
                shutil.copy(
                    label_path,
                    os.path.join(category_dir2, os.path.basename(label_path))
                )
            else:
                shutil.copy(
                    label_path,
                    os.path.join(category_dir1, os.path.basename(label_path))
                )
    import shutil
    label_names = os.listdir(label_dir)
    label_pathes = [os.path.join(label_dir, label_name) for label_name in label_names]
    start = 0
    pre_process = int(len(label_pathes) / process_num) + 1
    from multiprocessing import Process
    for i in range(process_num):
        end = start + pre_process
        if end > start:
            end = len(label_pathes)
        cur_pathes = label_pathes[start:end]
        process = Process(
            target=single_process,
            args=[cur_pathes, category_dir1, category_dir2]
        )
        process.start()
        start = end

'''
    提取patch 方法三
    针对每个联通分量
        1、我们提取整个ROI，将其放缩到指定的尺寸
        2、对Bounding Box使用滑窗的方法提取patch，ROI占比不得低于指定的阈值
'''
def extract_patch_method3(image_dir, mask_dir, patch_size, stride_size, occupy_rate, save_dir, process_num=8):
    def bounding_box(mask_image):
        xs, ys = np.where(mask_image != 0)
        return np.min(xs), np.max(xs), np.min(ys), np.max(ys)
    def single_process(image_pathes, mask_dir, patch_size, stride_size, occupy_rate, save_dir):
        for image_path in image_pathes:
            image = read_image(image_path)
            image = np.array(image)
            basename = os.path.basename(image_path).split('.tiff')[0]
            mask_pathes = glob(os.path.join(mask_dir, basename+'_*.png'))
            for index, mask_path in enumerate(mask_pathes):
                # 针对每一个联通分量
                count = 0
                mask_image = read_image(mask_path)
                mask_image = np.array(mask_image)
                if np.sum(mask_image == 255) <= 100:
                    print 'jump on: ', mask_path, ' sum is: ', np.sum(mask_image == 255)
                    continue
                # 不剔除该mask
                bounding = bounding_box(mask_image)
                category_patch1 = image[bounding[0]:bounding[1], bounding[2]: bounding[3]]
                save_image(
                    category_patch1,
                    os.path.join(save_dir, basename+'_' + str(index) + '_0.png')
                )
                count += 1
                for i in range(bounding[0], bounding[1], stride_size):
                    for j in range(bounding[2], bounding[3], stride_size):
                        cur_patch = image[i - patch_size / 2:i + patch_size / 2,
                                    j - patch_size / 2:j + patch_size / 2]
                        cur_mask = mask_image[i - patch_size / 2:i + patch_size / 2,
                                   j - patch_size / 2:j + patch_size / 2]
                        cur_rate = ((1.0 * np.sum(cur_mask != 0)) / (1.0 * patch_size * patch_size))
                        if cur_rate >= occupy_rate:
                            count += 1
                            save_path = os.path.join(save_dir,
                                                     basename + '_' + str(index) + '_' + str(i) + '_' + str(j) + '_'
                                                     + str(cur_rate) + '.png')
                            save_image(cur_patch, save_path)
                print mask_path, ' patch num: ', count
    image_names = os.listdir(image_dir)
    image_pathes = [os.path.join(image_dir, image_name) for image_name in image_names]
    start = 0
    per_process = len(image_pathes) / process_num + 1
    from multiprocessing import Process
    for i in range(process_num):
        end = start + per_process
        if end > len(image_pathes):
            end = len(image_pathes)
        cur_pathes = image_pathes[start: end]
        start = end
        process = Process(
            target=single_process,
            args=[cur_pathes, mask_dir, patch_size, stride_size, occupy_rate, save_dir,]
        )
        process.start()

'''
    提取patch 方法四
    针对每个联通分量
        1、我们提取整个ROI，将其放缩到指定的尺寸
        2、对Bounding Box使用滑窗的方法提取patch，ROI占比不得低于指定的阈值， 如果ROI等于1，则舍弃
'''
def extract_patch_method4(image_dir, mask_dir, patch_size, stride_size, occupy_rate, save_dir, process_num=8):
    def bounding_box(mask_image):
        xs, ys = np.where(mask_image != 0)
        return np.min(xs), np.max(xs), np.min(ys), np.max(ys)
    def single_process(image_pathes, mask_dir, patch_size, stride_size, occupy_rate, save_dir):
        for image_path in image_pathes:
            image = read_image(image_path)
            image = np.array(image)
            basename = os.path.basename(image_path).split('.tiff')[0]
            mask_pathes = glob(os.path.join(mask_dir, basename+'_*.png'))
            for index, mask_path in enumerate(mask_pathes):
                # 针对每一个联通分量
                count = 0
                mask_image = read_image(mask_path)
                mask_image = np.array(mask_image)
                if np.sum(mask_image == 255) <= 100:
                    print 'jump on: ', mask_path, ' sum is: ', np.sum(mask_image == 255)
                    continue
                # 不剔除该mask
                bounding = bounding_box(mask_image)
                category_patch1 = image[bounding[0]:bounding[1], bounding[2]: bounding[3]]
                save_image(
                    category_patch1,
                    os.path.join(save_dir, basename+'_' + str(index) + '_0.png')
                )
                count += 1
                for i in range(bounding[0], bounding[1], stride_size):
                    for j in range(bounding[2], bounding[3], stride_size):
                        cur_patch = image[i - patch_size / 2:i + patch_size / 2,
                                    j - patch_size / 2:j + patch_size / 2]
                        cur_mask = mask_image[i - patch_size / 2:i + patch_size / 2,
                                   j - patch_size / 2:j + patch_size / 2]
                        cur_rate = ((1.0 * np.sum(cur_mask != 0)) / (1.0 * patch_size * patch_size))
                        if cur_rate == 1.0:
                            continue
                        if cur_rate >= occupy_rate:
                            count += 1
                            save_path = os.path.join(save_dir,
                                                     basename + '_' + str(index) + '_' + str(i) + '_' + str(j) + '_'
                                                     + str(cur_rate) + '.png')
                            save_image(cur_patch, save_path)
                print mask_path, ' patch num: ', count
    image_names = os.listdir(image_dir)
    image_pathes = [os.path.join(image_dir, image_name) for image_name in image_names]
    start = 0
    per_process = len(image_pathes) / process_num + 1
    from multiprocessing import Process
    for i in range(process_num):
        end = start + per_process
        if end > len(image_pathes):
            end = len(image_pathes)
        cur_pathes = image_pathes[start: end]
        start = end
        process = Process(
            target=single_process,
            args=[cur_pathes, mask_dir, patch_size, stride_size, occupy_rate, save_dir,]
        )
        process.start()
'''
    提取某张图片的patch，然后保存下来
'''
def save_one_image_patches(image_path, save_dir):
    patches = extract_patchs_return(
        tiff_path=image_path,
        mask_dir=None,
        occupy_rate=None,
        stride=16,
        patch_size=256
    )
    for index, patch in enumerate(patches):
        save_image(patch,os.path.join(save_dir, str(index)+'.png'))
'''
    将文件夹下面的所有图片读取进来
'''
def read_images(image_dir):
    names = os.listdir(image_dir)
    images = [np.array(tiff_read(os.path.join(image_dir, name))) for name in names]
    return images
if __name__ == '__main__':
    # conver_all_svgs('/home/give/Game/GastricCancer/labels', '/home/give/Documents/dataset/BOT_Game/labels')
    # tiff_path = '/home/give/Documents/dataset/BOT_Game/train/negative/normal1.ndpi.16.5702_35104.2048x2048.tiff'
    # new_tiff_path = '/home/give/Documents/dataset/BOT_Game/512*512/train/negative/normal1.ndpi.16.5702_35104.2048x2048.tiff'
    # resize_tiff(tiff_path, new_tiff_path)
    # 可以放缩图片尺寸
    # resize_multi_dir(
    #     [
    #         '/home/give/Documents/dataset/BOT_Game/data_NY/train_resized/negative',
    #         '/home/give/Documents/dataset/BOT_Game/data_NY/train_resized/positive',
    #         '/home/give/Documents/dataset/BOT_Game/data_NY/original_resized/negative',
    #         '/home/give/Documents/dataset/BOT_Game/data_NY/original_resized/positive'
    #     ],
    #     [224,   224]
    # )

    # extract_patch_single_tiff(
    #     '/home/give/Documents/dataset/BOT_Game/train/positive/2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048.tiff',
    #     '/home/give/Documents/dataset/BOT_Game/labels/2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048.svg.png',
    #     '/home/give/PycharmProjects/StomachCanner/tools/patches.npy',
    #     patch_size=256,
    #     stride_size=32
    # )
    # fill_region('/home/give/Documents/dataset/BOT_Game/labels/2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048.svg.png')
    # 提取所有的ｐａｔｃｈ, 按照ｓｃｏｒｅ和阈值打ｌａｂｅｌ
    # tiff_save_paths = [
    #     '/home/give/Documents/dataset/BOT_Game/train/positive',
    #     '/home/give/Documents/dataset/BOT_Game/train/negative',
    #     '/home/give/Documents/dataset/BOT_Game/val/positive',
    #     '/home/give/Documents/dataset/BOT_Game/val/negative',
    # ]
    # patches_save_paths = [
    #     '/home/give/Documents/dataset/BOT_Game/patches/256-32/train/records.txt',
    #     '/home/give/Documents/dataset/BOT_Game/patches/256-32/train/records.txt',
    #     '/home/give/Documents/dataset/BOT_Game/patches/256-32/val/records.txt',
    #     '/home/give/Documents/dataset/BOT_Game/patches/256-32/val/records.txt',
    # ]
    # patches_save_image_paths = [
    #     '/home/give/Documents/dataset/BOT_Game/patches/256-32/train',
    #     '/home/give/Documents/dataset/BOT_Game/patches/256-32/train',
    #     '/home/give/Documents/dataset/BOT_Game/patches/256-32/val',
    #     '/home/give/Documents/dataset/BOT_Game/patches/256-32/val',
    # ]
    # png_path = '/home/give/Documents/dataset/BOT_Game/fill_label'
    # extract_patches_multi_dir(
    #     tiff_save_paths,
    #     patches_save_paths,
    #     png_path,
    #     patches_save_image_paths
    # )

    # 提取所有绝对是癌症区域的ｐａｔｃｈ
    # pngs_dir = '/home/give/Documents/dataset/BOT_Game/fill_label'
    # tiffs_dirs = [
    #     '/home/give/Documents/dataset/BOT_Game/train/positive',
    #     '/home/give/Documents/dataset/BOT_Game/val/positive',
    # ]
    # save_patch_dirs = [
    #     '/home/give/Documents/dataset/BOT_Game/patches/absolute/train/positive',
    #     '/home/give/Documents/dataset/BOT_Game/patches/absolute/val/positive'
    # ]
    # for index, tiffs_dir in enumerate(tiffs_dirs):
    #     extract_absolution_positive_dirs(
    #         tiffs_dir,
    #         pngs_dir,
    #         save_patch_dirs[index],
    #         stride_size=80,
    #         patch_size=256
    #     )


    # 提取绝对是正常区域的ｐａｔｃｈ
    # pngs_dir = None
    # tiffs_dirs = [
    #     '/home/give/Documents/dataset/BOT_Game/train/negative',
    #     '/home/give/Documents/dataset/BOT_Game/val/negative',
    # ]
    # save_patch_dirs = [
    #     '/home/give/Documents/dataset/BOT_Game/patches/absolute/train/negative',
    #     '/home/give/Documents/dataset/BOT_Game/patches/absolute/val/negative'
    # ]
    # for index, tiffs_dir in enumerate(tiffs_dirs):
    #     extract_absolution_positive_dirs(
    #         tiffs_dir,
    #         pngs_dir,
    #         save_patch_dirs[index],
    #         stride_size=256,
    #         patch_size=256
    #     )

    # 计算平均图像
    # calu_average_train_set('/home/give/Documents/dataset/BOT_Game/patches/absolute/train/')
    # convert_images_type('/home/give/Documents/dataset/BOT_Game/512*512/train/negative', 'png')

    # print has_same_file('/home/give/Documents/dataset/BOT_Game/data_NY/original_jpg/positive', '/home/give/Documents/dataset/BOT_Game/data_NY/patch/train/positive')

    # extract_patches_one_folder(
    #     tiff_dir='/home/give/Documents/dataset/BOT_Game/val/positive-histeq',
    #     mask_dir='/home/give/Documents/dataset/BOT_Game/fill_label',
    #     patch_size=256,
    #     stride=128,
    #     occupy_rate=0.5,
    #     save_dir='/home/give/Documents/dataset/BOT_Game/patches/256-all-histeq/val/positive'
    # )
    extract_patches_one_folder_num(save_dir='/home/give/Documents/dataset/BOT_Game/patches/method4/val/negative',
                                   patch_size=256,
                                   all_num=2200,
                                   tiff_dir='/home/give/Documents/dataset/BOT_Game/val/negative')

    # delete_positive_pathc(
    #     '/home/give/Documents/dataset/BOT_Game/patches/256-delete-patch/val/positive',
    #     0.8
    # )
    # convert_format(
    #     '/home/give/Documents/dataset/BOT_Game/train/negative',
    #     '/home/give/Documents/dataset/BOT_Game/train/negative-png',
    #     process_num=10
    # )
    # 1116 423 -> 1115 423
    # split_connected_region('/home/give/Documents/dataset/BOT_Game/fill_label/2017-06-10_17.15.00.ndpi.16.35008_20557.2048x2048.png', '/home/give/Documents/dataset/BOT_Game/fill_label_splited')
    # split_connected_regions(
    #     '/home/give/Documents/dataset/BOT_Game/fill_label',
    #     '/home/give/Documents/dataset/BOT_Game/fill_label_splited'
    # )

    # extract_patches_single_connected_regions(
    #     mask_dir='/home/give/Documents/dataset/BOT_Game/fill_label_splited',
    #     image_dirs=[
    #         '/home/give/Documents/dataset/BOT_Game/train/positive',
    #         '/home/give/Documents/dataset/BOT_Game/val/positive'
    #     ],
    #     save_dirs=[
    #         '/home/give/Documents/dataset/BOT_Game/patches/256-singleconnected-patch/train/positive',
    #         '/home/give/Documents/dataset/BOT_Game/patches/256-singleconnected-patch/val/positive'
    #     ],
    #     patch_size=256,
    #     stride_size=128,
    #     occury_rate=0.5,
    # )

    # calu_static_areas(
    #     '/home/give/Documents/dataset/BOT_Game/fill_label_category/cell1'
    # )

    # splited_labels_into_categorys(
    #     '/home/give/Documents/dataset/BOT_Game/fill_label_splited',
    #     '/home/give/Documents/dataset/BOT_Game/fill_label_category/cell',
    #     '/home/give/Documents/dataset/BOT_Game/fill_label_category/region'
    # )

    # extract_patch_method3(
    #     image_dir='/home/give/Documents/dataset/BOT_Game/val/positive',
    #     mask_dir='/home/give/Documents/dataset/BOT_Game/fill_label_splited',
    #     save_dir='/home/give/Documents/dataset/BOT_Game/patches/method3/val/positive',
    #     occupy_rate=0.5,
    #     patch_size=256,
    #     stride_size=128,
    #     process_num=8
    # )
    # save_one_image_patches(
    #     '/home/give/Documents/dataset/BOT_Game/train/positive/2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048.tiff',
    #     '/home/give/Documents/dataset/BOT_Game/train/positive-test'
    # )

    # extract_patch_method4(
    #     image_dir='/home/give/Documents/dataset/BOT_Game/val/positive',
    #     mask_dir='/home/give/Documents/dataset/BOT_Game/fill_label_splited',
    #     save_dir='/home/give/Documents/dataset/BOT_Game/patches/method4/val/positive',
    #     occupy_rate=0.6,
    #     patch_size=256,
    #     stride_size=128,
    #     process_num=8
    # )