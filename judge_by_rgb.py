volumeNum = 100
scaling = 1000
path = 
'/home/sunliqun/git/MitosisDetection/src/backend/data/camelyon/train'
# path 是图片的路径

slide = open_slide(path + '/Tumor_005.tif')
(x, y) = slide.dimensions
Multiple = x / scaling
yM = y * Multiple
slide_thumbnail = slide.get_thumbnail((scaling, yM))
slide_thumbnail.save('0023.tif')

slide = open_slide('0023.tif')
(x, y) = slide.dimensions

RGBListPortrait = list()
RGBListTransverse = list()
varListPortrait = list()
varListTransverse = list()

for i in range(0, y, volumeNum):
    arr = array(slide.read_region((0, i), 0, (x, volumeNum)))
    arrR = np.mean(arr[:, :, :1])
    arrG = np.mean(arr[:, :, 1:2])
    arrB = np.mean(arr[:, :, 2:3])
    RGBListPortrait.append((arrR, arrG, arrB))
for i in range(0, x, volumeNum):
    arr = array(slide.read_region((i, 0), 0, (volumeNum, y)))
    arrR = np.mean(arr[:, :, :1])
    arrG = np.mean(arr[:, :, 1:2])
    arrB = np.mean(arr[:, :, 2:3])
    RGBListTransverse.append((arrR, arrG, arrB))

for i, rgbVar in enumerate(RGBListPortrait):
    RGBSpot = np.var(rgbVar)
    if RGBSpot >= 1:
        varListPortrait.append(i)
for i, rgbVar in enumerate(RGBListTransverse):
    RGBSpot = np.var(rgbVar)
    if RGBSpot >= 1:
        varListTransverse.append(i)

startX = min(varListTransverse) * volumeNum
startY = min(varListPortrait) * volumeNum
width = ((max(varListTransverse) + 1) - (min(varListTransverse))) * 
volumeNum
height = ((max(varListPortrait) + 1) - (min(varListPortrait))) * 
volumeNum

print(startX*Multiple, startY*Multiple)
print(width*Multiple, height*Multiple)
