"""
将xml标签转换为txt格式，自动划分训练集和验证集。
"""
import os.path
import xml.etree.ElementTree as ET
import random
from shutil import copyfile


classes=["car","person"]
TRAIN_RATIO=80

def convert(size,bbox):
    # 输入图片的size和bndbox的坐标，转换为(x,y,w,h)的元组,表示bndbox在图片中的中心x,y坐标和对应的x，y的宽和高
    # 获取图片size,并归一化
    width=1.0/size[0]
    height=1.0/size[1]
    # 获取bndbox中心x,y坐标
    x=(bbox[0]+bbox[2])/2.0
    y=(bbox[1]+bbox[3])/2.0
    w=bbox[2]-bbox[0]  # x坐标值相减得到bbox的宽
    h=bbox[3]-bbox[1]  # y坐标值相减得到bbox的高
    # 归一化
    x=x*width
    y=y*height
    w=w*width
    h=h*height
    return (x,y,w,h)


def convert_annotation(image_id):
    # 打开一张xml文件
    in_file=open("VOCdevkit/VOC2007/Annotations/{}.xml".format(image_id))
    # 以写模式输出一张txt文件
    out_file=open("VOCdevkit/VOC2007/YOLOLabels/{}.txt".format(image_id),'w')
    # 将输入的xml文件解析为树模型
    tree=ET.parse(in_file)
    # 从树模型获得根目录
    root=tree.getroot()
    # 从根目录中获得size标签目录
    size=root.find('size')
    # 从size中获得宽width标签和高height标签，并取到其中的text内容，并将其变成int类型
    width=int(size.find('width').text)
    height=int(size.find('height').text)

    # 观察xml文件得知，在一张图片的xml中的size标签目录只有一个，而一张图片中的object标签目录有可能有多个，故而要循环迭代获取xml中的object标签目录
    # 当你用xml格式的iter()方法去寻找object标签会得到一个包含所有object标签列表，是一个可迭代的对象。注意：这里使用find()方法只能寻找到第一个object标签
    # root.iter()返回一个迭代器，然后用for循环遍历这个迭代器，每次遍历一个object标签目录
    for obj in root.iter('object'):
        difficult=obj.find('difficult').text  # 提取difficult标签内容
        clas=obj.find('name').text    # 提取类名
        if clas not in classes or difficult==1:
            continue
        bbox=obj.find('bndbox')  # 提取bndbox标签
        b=(float(bbox.find('xmin').text),float(bbox.find('ymin').text),float(bbox.find('xmax').text),
           float(bbox.find('ymax').text))   # 提取bodbox坐标
        b_convert=convert((width,height),b)  # 将bodbox坐标完成转换成yolo格式的(x,y,w,h)的元组
        clas_id=classes.index(clas)  # 输入类名，得到对应索引
        out_file.write(str(clas_id)+" "+" ".join([str(a) for a in b_convert])+"\n")  # 类名+空格字符+列表生成式通过join生成字符串 ，加上换行符
    # 关闭打开的open
    in_file.close()
    out_file.close()


def create_dir(cur_dir,dir_name):
    # 创建目录
    new_dir=os.path.join(cur_dir,dir_name)  # 将当前目录和目录名组成新目录
    if not os.path.isdir(new_dir):   # 如果目录不存在，创建这个目录
        os.mkdir(new_dir)
    return new_dir


cwd=os.getcwd()  # 获取当前工作目录
VOCdevkit_dir=create_dir(cwd,"VOCdevkit/")  # 生成VOCdevkit工作目录

images_dir=create_dir(VOCdevkit_dir,"images/")  # 生成images工作目录
images_train_dir=create_dir(images_dir,"train/")
images_val_dir=create_dir(images_dir,"val/")

labels_dir=create_dir(VOCdevkit_dir,"labels/")  # 生成labels工作目录
labels_train_dir=create_dir(labels_dir,"train/")
labels_val_dir=create_dir(labels_dir,"val/")

VOC2007_dir=create_dir(VOCdevkit_dir,"VOC2007/")  # 生成VOC2007工作目录
Annotations_dir=create_dir(VOC2007_dir,"Annotations/")  # 生成Annotations工作目录。注意：这个目录应该提取存在。
JPEGImages_dir=create_dir(VOC2007_dir,"JPEGImages/")  # 生成JPEGImages工作目录。注意：这个目录应该提取存在。
YOLOLabels_dir=create_dir(VOC2007_dir,"YOLOLabels/")  # 生成YOLOLabels工作目录。

# 首先创建两个txt文件，以写入模式打开,来记录图片保存的路径，然后获取JPEGImg文件夹中所有文件名，
# 遍历所有文件名，将文件名和文件目录组成一条完整的图片路径，检查图片路径是否存在，如果文件路径存在，给个内存存储文件名，
# 将JPEGImg分割成img的文件名和扩展名，使用文件名生成xml路径和yololabel路径。目的检查xml路径是否存在。
# 生成一个概率值，依次划分训练集和验证集。依据概率值,将转换完成的yololabel路径复制到训练或验证集，同时也将对应的图片复制到训练集或验证集

train_file=open(os.path.join(cwd,"yolov5_train.txt"),'w')  # 创建txt记录保存的训练集img路径
val_file=open(os.path.join(cwd,"yolov5_val.txt"),'w')   # 创建txt记录保存的验证集img路径
list_JPEGImgs=os.listdir(JPEGImages_dir)  # 得到JPEGImg文件夹下的文件名，获取JPEGImages目录下所有图片名列表.如[001.jpg,002.jpg]
# 遍历每个图片名
for i in range(len(list_JPEGImgs)):
    JPEGImages_path=os.path.join(JPEGImages_dir,list_JPEGImgs[i])  # 得到一张图片文件，JPEG目录和一张图片名组合成一张图片的路径，如“000000.jpg
    if os.path.isfile(JPEGImages_path):  # 如果图片文件路径存在
        voc_path=list_JPEGImgs[i]  # 存储这个文件名，获取图片名
        (nameWithoutExtention,extention)=os.path.splitext(os.path.basename(JPEGImages_path))  # 将一张图片的path分割成文件名和扩展名。返回最后的文件名，图片名和后缀名分开，
        Annotations_name = nameWithoutExtention+".xml"   # 得到xml扩展名的文件
        Annotations_path = os.path.join(Annotations_dir,Annotations_name)  # 生成Anno路径，组成完整标注名,Anno 和xml文件组成xml路径
        YOLOLabels_name = nameWithoutExtention+".txt"  # 得到txt扩展名的文件
        YOLOLabels_path = os.path.join(YOLOLabels_dir,YOLOLabels_name)  # 生成yololabel路径，等格式转换后会将这个路径下的yololabel复制到训练集或验证集。组成完整yolo标签名，yololabels 和txt组成txt路径
    prob=random.randint(1,100)  # 随机生成一个1~100之间的整数，作为概率值
    print("prob:{}".format(prob))
    if os.path.exists(Annotations_path):  # 如果xml标签存在
        if prob<TRAIN_RATIO:  # 训练比例小于80，作为训练集
            train_file.write(JPEGImages_path+'\n')   # 将图片路径写入train_file，一个train.txt
            convert_annotation(nameWithoutExtention)  # 输入文件名，将xml转换成yolo，并存在yololabel文件夹。输入没有扩展名的图片名字，通过函数xml标签转换成yolo标签保存到yololabels文件夹
            copyfile(JPEGImages_path,images_train_dir+voc_path)  # 将JPEGImg复制到images/train/下，并重命名为voc_path
            copyfile(YOLOLabels_path,labels_train_dir+YOLOLabels_name)  # 将转换好的yololabel复制到labels/train/下，并重命名为YOLOLabels_name
        else:  # 如果随机值大于80，作为验证集
            val_file.write(JPEGImages_path+'\n')
            convert_annotation(nameWithoutExtention)
            copyfile(JPEGImages_path,images_val_dir+voc_path)
            copyfile(YOLOLabels_path,labels_val_dir+YOLOLabels_name)  # 将生成的yololabels图片复制到图片验证集
train_file.close()
val_file.close()

















