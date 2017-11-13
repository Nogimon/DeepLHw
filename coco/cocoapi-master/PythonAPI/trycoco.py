from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import cv2

#set parameters
dire = '/media/zlab-1/Data/Lian/course/DeepLHw/coco/'
dataDir=dire
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dire,dataType)

#load coco api and coco categories
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

#load data
catIds = coco.getCatIds(catNms = ['person'])
imgIds = coco.getImgIds(catIds = catIds)
img = coco.loadImgs(imgIds)

#annIds = coco.getAnnIds(catIds = catIds)
#anns = coco.loadAnns(imgIds)

train = []
y = []
for i in range(100):#len(img)):
    I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img[i]['file_name']))
    I2 = cv2.resize(I, (128, 128), cv2.INTER_LINEAR)
    
    annIds = coco.getAnnIds(imgIds=img[i]['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    mask = coco.annToMask(anns[0])
    mask2 = cv2.resize(mask, (128, 128), cv2.INTER_LINEAR)
    
    I2 = np.asarray(I2)
    mask2 = np.asarray(mask2)
    train.append(I2)
    y.append(mask2)
    
train = np.asarray(train)
y = np.asarray(y)


    
    
    

    


'''
dire = '/media/zlab-1/Data/Lian/course/DeepLHw/coco/'
dataDir=dire
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dire,dataType)

coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

img = coco.loadImgs(catIds = catIds)
img = coco.loadImgs(imgIds)

# load and display image
I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()


# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)


ann = anns[2]
polygons = []
for seg in ann['segmentation']:
    poly = np.array(seg).reshape((int(len(seg)/2), 2))
    polygons.append(Polygon(poly))
plt.plot(poly[:,0],poly[:,1])

for ann1 in anns:
    mask = coco.annToMask(ann1)
    plt.figure()
    plt.imshow(mask)

'''

