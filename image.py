import numpy as np
import cv2 
gray_imgs = cv2.imread('gray_0.png')
#gray_imgs = cv2.cvtColor(gray_imgs, cv2.COLOR_BGR2GRAY)
import torch
from skimage import img_as_float
im1=[]
for i in range(1):
   # im= cv2.cvtColor(gray_imgs, cv2.COLOR_BGR2GRAY)
    im1.append(np.array(img_as_float(gray_imgs)))
img1 = np.array([im1])
imd = torch.from_numpy(img1) #[:,:,:,1])
imd.shape
# perform automatic thresholding
import skimage
blurred_image = skimage.filters.gaussian(gray_imgs, sigma=1.0)
t = skimage.filters.threshold_otsu(blurred_image)
ret, bw_img = cv2.threshold(gray_imgs, 80, 255, cv2.THRESH_BINARY)  
# converting to its binary form
#bw = cv2.threshold(gray_imgs, 127, 255, cv2.THRESH_BINARY)
import matplotlib.pyplot as plt  
plt.imshow(bw_img)

# detecting the blob
sim  = gray_imgs.copy()
#for j in range(1): 
    #print('j=', j)
blobs_doh = blob_doh(gray_imgs, min_sigma=10,  max_sigma=20, threshold=.001, overlap=0.8)
fig, ax = plt.subplots()
circx=[];circy=[];circr=[];n1c=[];circa=[]
#sim = imgs[j].copy()
#sim1 = Image.open('Downloads/shubhangi/shubh/gray/gray_0.png')
sim1 = cv2.imread('./gray_0.png') 
bl = 0
for blob in blobs_doh:
   # print('blob=', blob)
    nc = j
    bl +=1
    yc, xc, rc = blob
    area = pi*math.pow(rc,2)
    circx.append(xc); circy.append(yc); circr.append(rc); circa.append(area)
    n1c.append(nc)
    #draw = ImageDraw.Draw(sim)
    #draw.ellipse((xc-rc,yc-rc,xc+rc,yc+rc),fill= 'red')
    sim1 = cv2.circle(sim1, (int(xc),int(yc)), int(rc), color=(0,0,255), thickness=1)
    sim1 = cv2.putText(sim1, text=str(bl), org=(int(xc), int(yc)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale =0.2,color=(0, 255, 0),thickness=1)
   # print(os.getcwd())
    cv2.imwrite('./shubh2/circ/b_{}.png'.format(j+1), sim1)
    #sim.save('Downloads/shubhangi/shubh/circle/DoH_{}.png'.format(j))
z11 = np.column_stack([circx, circy, circr, circr, circa])
dfz11 = pd.DataFrame(z11, columns=['circ_x', 'circ_y', 'Radius', 'Radius2', 'Area'])                           #.to_csv('Downloads/shubhangi/center/c_'+str(j)+'.csv')
cd ..
# trying to look for superpixel that belong to the blob
for j in range(1):
#First SLIC segmentation of imgs[j]
    segments_slic = slic(sim, n_segments=20000, compactness=10, sigma=1, start_label=1)
    '''
    for index in np.ndindex(segments_slic.shape):
        print('current = ', segments_slic[index])
        print('cur = ', index)
    '''
    segments_ids = np.unique(segments_slic)
    
    id_max = np.max(segments_ids)
    #print('id= ', id_max)

    superpixel_list = sp_idx(segments_slic)
    superpixel = [idx for idx in superpixel_list]
   

    reimg = mark_boundaries(gray_imgs, segments_slic)
    #cv2.imwrite('Downloads/re/remask_'+str(j)+'.png', mark_boundaries(reimg, segments_slic))
    #Calculating properties of each superpixel
    x=[0 for i in range(len(superpixel))]
    #print('length=', superpixel[15] )
    y=[0 for i in range(len(superpixel))] 
    #centers = np.array([np.mean(np.nonzero(segments_slic == i), axis=1) for i in segments_ids])
    w = []; h = []; rad=[];rad2=[];centx = []; centy = [];im_no=[];X_dataset=[]; minr =[]
    sp_xmin = []; sp_xmax = []; sp_ymin = []; sp_ymax = []
    sp_xl=[];sp_yl=[];cex=[];cey=[]

    #First SLIC segmentation of imgs[j]
    segments_slic2 = slic(gray_imgs, n_segments=1800, compactness=10, sigma=1, start_label=1)
    segments_ids2 = np.unique(segments_slic2)
    id_max2 = np.max(segments_ids2)
    #print('id2= ', id_max2)
    superpixel_list2 = sp_idx(segments_slic2)
    superpixel2 = [idx for idx in superpixel_list2]
    reimg2 = mark_boundaries(gray_imgs, segments_slic2)
    #cv2.imwrite('Downloads/re/remask2_'+str(j)+'.png', mark_boundaries(reimg, segments_slic2)*255)
    #Calculating properties of each superpixel
    x2=[0 for i in range(len(superpixel2))]
    y2=[0 for i in range(len(superpixel2))] 

    #centers = np.array([np.mean(np.nonzero(segments_slic == i), axis=1) for i in segments_ids])
    w2 = []; h2 = []; rad3=[];rad4=[]; centx2 = []; centy2 = []; im_no2=[];  X_dataset2 = []; minr2 =[]; maxr2=[]; fiii =[]; fii1=[]; sp_lx=[];sp_ly=[]; sp_dist=[]
    sp_x2 = []; sp_y2 = []; sp_ux2 = []; sp_uy2 = []; sp_xmin2 = []; sp_xmax2 = []; sp_ymin2 = []; sp_ymax2 = []; fcentx=[]; fcenty=[]; fi =[]; fi1 =[];fi2=[];fi3=[]
    sp_area2=[]
    for segVal in np.unique(segments_slic):           #segval is im_sp_centroid=[] 1,2,3,4,5,6,7,8,9 i.e. superpixels
#
        #creating a mask and saving each segment in a folder mask2
        mask = np.ones(gray_imgs.shape[:2], dtype='uint8') #   self.height, self.width = img.shape[:2]
        mask[segments_slic == segVal] = 255
        pos = np.where(mask == 255)
        #properties of each superpixel
        x = pos[:][0]  #  XY = np.array([superpixel[i][0], superpixel[i][1]]).T
        y = pos[:][1]
        #print('x=', x)
        lx1 = list(x);ly1=list(y)
        lisxy = [(x,y) for x,y in zip(lx1,ly1)]
        sp_xl.append(lisxy)
        len_x = len(x); len_y = len(y)
        ymin = np.min(pos[:][1]); ymax = np.max(pos[:][1])
        xmin = np.min(pos[:][0]); xmax = np.max(pos[:][0])
        sp_xmin.append(xmin); sp_xmax.append(xmax)
        sp_ymin.append(ymin); sp_ymax.append(ymax)
        cx = np.mean(x); cy = np.mean(y)
        liscxy =(cx,cy)
        sp_yl.append(liscxy)
        width = xmax - xmin + 1; w.append(width)
        height = ymax - ymin + 1; h.append(height)
        radius = width/2; rad.append(radius)
        radius2 =height/2; rad2.append(radius2)
        minrad = min(int(radius), int(radius2))
        minr.append(minrad)
        sp_x.append(x); sp_y.append(y)
        centx.append(int(cx)); centy.append(int(cy))
        cex.append(cx);cey.append(cy)
        im_no.append(j)
        
        
    regions2 = measure.regionprops(segments_slic2, intensity_image=gray_imgs)
    for r in regions2:
        area2 = r.area
        sp_area2.append(area2)

    for segVal2 in np.unique(segments_slic2):           #segval is im_sp_centroid=[] 1,2,3,4,5,6,7,8,9 i.e. superpixels
#
        #creating a mask and saving each segment in a folder mask2
        mask2 = np.ones(gray_imgs.shape[:2], dtype='uint8') #   self.height, self.width = img.shape[:2]
        mask2[segments_slic2 == segVal2] = 255
        pos2 = np.where(mask2 == 255)
        #properties of each superpixel
        x2 = pos2[:][0]  #  XY = np.array([superpixel[i][0], superpixel[i][1]]).T
        y2 = pos2[:][1]
        lx2 = list(x2); ly2= list(y2)
        #  i want to calculate center for nearly circular superpixel by finding out
        # maximum distance between two pairs and then getting mid point
        lisxy = [(x,y) for x,y in zip(lx2,ly2)]
       
        fiii.append(lisxy)
        sp_lx.append(lx2),sp_ly.append(ly2)
        dist = distance.cdist(np.array(lisxy), np.array(lisxy), 'euclidean')        
        fi = np.max(dist)
        fi1 = np.where(dist ==fi)
        sp_dist.append(fi/2)
       
        fi3.append(fi1)                 
        len_x2 = len(x2); len_y2 = len(y2)
        ymin2 = np.min(pos2[:][1]); ymax2 = np.max(pos2[:][1])
        xmin2 = np.min(pos2[:][0]); xmax2 = np.max(pos2[:][0])
        sp_xmin2.append(xmin2); sp_xmax2.append(xmax2)
        sp_ymin2.append(ymin2); sp_ymax2.append(ymax2)
        cx2 = np.mean(x2); cy2 = np.mean(y2)
        width2 = xmax2 - xmin2 + 1; w2.append(width2)
        height2 = ymax2 - ymin2 + 1; h2.append(height2)
        radius3 = width2/2; rad3.append(radius3)
        radius4 = height2/2; rad4.append(radius4)
        minrad2 = min(int(radius3), int(radius4))
        minr2.append(minrad2)
        maxrad2 = max(int(radius3), int(radius4))
        maxr2.append(maxrad2)
        sp_ux2.append(list(np.unique(x2))); sp_uy2.append(list(np.unique(y2)))
        sp_x2.append(x2); sp_y2.append(y2)
        centx2.append(cx2); centy2.append(cy2)
        im_no2.append(j)
    dfi = pd.DataFrame(fi3).to_csv('./gray/shubh2/ind2/i_'+str(j)+'.csv', sep=',', index=True, header=True)
  #  dfi1 = np.column_stack([sp_lx, sp_ly,sp_dist])
    #dfi2 = pd.DataFrame(dfi1, columns=['LX','LY','Radius']).to_csv('./gray/shubh2/lisxy2/i_'+str(j)+'.csv', sep=',', index=True, header=True)
    
    '''
    v = np.column_stack([segments_ids, sp_y, sp_x, centy, centx,cey,cex, minr, w, h])
    dfv = pd.DataFrame(v, columns=['sp_ID', 'X', 'Y', 'cent_X', 'cent_Y','CX','CY', 'minr', 'width', 'height']).to_csv('./gray/shubh2/data2/im_'+str(j)+'_data.csv', sep=',', index=False, header=True)

    v2 = np.column_stack([segments_ids2, sp_y2, sp_x2, centy2, centx2, rad3, rad4, minr2, maxr2, w2, h2, sp_ux2, sp_uy2])
    dfv2 = pd.DataFrame(v2, columns=['sp_ID', 'X', 'Y','cent_X', 'cent_Y', 'rad3', 'rad4', 'minr2', 'maxr2', 'width', 'height', 'UX', 'UY']).to_csv('./gray/shubh2/data1/im_'+str(j)+'_data.csv', sep=',', index=False, header=True)
    
    v3 = np.column_stack([centy2, centx2, minr2, maxr2, sp_area2])
    dfz12 = pd.DataFrame(v3, columns=['circ_x', 'circ_y', 'Radius','Radius2','Area'])                                  #.to_csv('Downloads/shubhangi/center1/im_'+str(j)+'_data.csv', sep=',', index=True, header=True)

    
    b1 = pd.read_csv('./gray/shubh2/data2/im_'+str(j)+'_data.csv')
    b2 = pd.read_csv('./gray/shubh2/data1/im_'+str(j)+'_data.csv')
    '''
    v3 = np.column_stack([centy2, centx2, minr2, maxr2, sp_area2])
    dfz12 = pd.DataFrame(v3, columns=['circ_x', 'circ_y', 'Radius','Radius2','Area'])  
    def Union(lst1, lst2):
        final_list = lst1 + lst2
        return final_list
        
    def PointsInCircum(ep,i,n):
        centerx = [(centx[i]+math.cos(pi/n)*ep,b1.centy[i]+math.sin(pi/n)*ep)]                     # +math.cos(pi/n)*ep
        return centerx
    
    def circle(x1, y1, x2, y2, r1, r2):
  
        distSq = (((x1 - x2)* (x1 - x2))+ ((y1 - y2)* (y1 - y2)))**(.5)
        if (distSq<=r2):
            #print('lies inside =', distSq)
            return r2

        
    ap1=[]; bp1=[];ap2=[]; bp2=[];out=[]; cent =[];circp=[]
    #eps = b1.minr.values; eps2 = b2.minr2.values
    ps = minr; eps2 = minr2
    k1=0;ii = []; d3=[]; d4=[];d5=[];d6=[]; indx=[];indx1=[];ii1=[];k=0;j1=[];b10=[];b11=[];ex=[];ey=[]
    circ_i = []
# circle filter se subtract kiya 1000 wale data ko to get center and exact superpixel coordinates then from there 10 wale k x and y coordinates mn dekha ki center h ya nahi 
    for i, jj in zip(dfz11.itertuples(index = False, name ='Pandas'), range(len(dfz11))):
        #print('fiest time j_jj=', str(j)+'_'+str(jj))
        #print('i=', i)
        a3 = dfz12.subtract(i, axis=1)
        ax  = pd.DataFrame(a3).to_csv('./gray/shubh2/comparison/t_'+str(j)+'_'+str(jj)+'.csv', sep=',', index = True, header=True)
        a1 = pd.read_csv('./gray/shubh2/comparison/t_'+str(j)+'_'+str(jj)+'.csv')        

        for index1, i2 in zip(range(len(a1)), a1.iterrows()):
            k1 =  k1+1
         
            if((0<=abs(a1.circ_x[index1])<=2.5 and 0<=abs(a1.circ_y[index1])<=2.5)  and (abs(a1.Radius[index1])<=10 and abs(a1.Radius2[index1])<=10)  ): 
                print('r1=',  abs(a1.Radius[index1]), 'r2=',  abs(a1.Radius2[index1]))
                
          
                if os.path.isfile('./gray/shubh2/im1/mask_'+str(j)+'_'+str(index1)+'.png'):  
                    print('t_'+str(j)+'_'+str(jj)+'')
                    print('index1=', index1)
                    read_image = Image.open('./gray/shubh2/im1/mask_'+str(j)+'_'+str(index1)+'.png')
                    read_image.save('./gray/shubh2/read1/mask_'+str(j)+'_'+str(index1)+'.png')
              
                k = k+1
                circ_i.append(jj)
                indx.append(index1)
                #print('index=', len(indx))
                ii.append(i2)
                b3 = sp_y[index1]; c3 = sp_x[index1]
                b6 = centy[index1]; c6 = centx[index1]
                b4 = segments_ids2#b2['sp_ID']
                #print('b=', len(indx))
                b5 = centy; c5 = centx
                #bc5 = CX; b5 = CY.
                #print('pair = ',(b6,c6))
                element =  [l for l in range(len(b5)) if str(b5[l]) in b3]
                element2 = [l for l in range(len(b5)) if str(c5[l]) in c3] 
                union_list = np.unique(Union(element,element2))
                #eps1 = eps[index1]
                ind1 = []
                for po in union_list:
                   
                    #ax4 = b1.LX[po]; ay4 = b1.LY[po]
                    ax4 = sp_xl[po]
                 #   eps1 = eps[po]
                  #  cir = PointsInCircum(eps1,po,4)
                    #print('cir=', cir)
               #     poo = 0
               #     for pai in range(len(cir)):
                    x1, y1 = b5[po], c5[po]                          #cir[pai]
                            #print('x1=', x1, 'y1=', y1)
                    
                    x2 ,y2 = b6, c6
                  #  x3, y3 = bc5[po],cb5[po]
                    #print('x2=', x2, 'y2=', y2)
                    r1 ,r2 = eps[po],eps2[index1]
                    #print('r1=', r1); print('r2=', r2)
                    output = circle(x1, y1, x2, y2, r1, r2)
                    if output == r2:
                    
                        ex.append(ax4)              #;ey.append(ay4)
                        ap1.append(x3)
                        bp1.append(y3)
                        ap2.append(x1)
                        bp2.append(y1)
                        out.append(po)
                        #poo +=1
                        r1 = Image.open('./gray/hubh2/m2/mask_'+str(j)+'_'+str(po)+'.png')
                        r1.save('./gray/shubh2/read3/mask_'+str(j)+'_'+str(po)+'.png')
     
                        if os.path.isfile('./gray/shubh2/im2/mask_'+str(j)+'_'+str(po)+'.png'):
                           
                            out.append(po)
                            read_image2 = Image.open('./gray/shubh2/im2/mask_'+str(j)+'_'+str(po)+'.png')
                            read_image2.save('./gray/shubh2/read3/mask_'+str(j)+'_'+str(po)+'.png')

                out1 = pd.DataFrame(ex).to_csv('./gray/shubh2/ex2/mask_'+str(j)+'.csv')
                out2 = pd.DataFrame(out).to_csv('./gray/shubh2/out2/mask_'+str(j)+'.csv')

        ad1 = pd.DataFrame(ap1).to_csv('./gray/shubh2/ap2/ap_'+str(j)+'.csv')
        bd1 = pd.DataFrame(bp1).to_csv('./gray/shubh2/bp2/bp_'+str(j)+'.csv')
        
# giving gray color to unwanted blobs to make image processing easier 
for i in circ_i:
    sim1 = cv2.circle(sim1, (int(circy[i]+5),int(circx[i]+5)), int(circr[i]+5), color=(38,38,38), thickness=-1)
    cv2.imwrite('./shubh2/circ/b_{}.png'.format(j+3), sim1)

# highlighting blobs that are of interest 
data = cv2.imread('./gray/gray_0.png') #.#np.zeros((932,932,3), dtype=np.uint8)       #    32x32 patch 
#data[0:931, 0:931] = [0,0,0] #[255,128,0]
# sp_x2 is for 1800 segmentation 
for i in indx:
    data[sp_x2[i], sp_y2[i]] = [255,128,0] # np.array(img1)
img4 = Image.fromarray(data)
img4.save('./gray/shubh2/circ/re8.png')
'''
# total length of circle 
list1 = list(np.arange(0,296,1))
# remove detected superpixels 
list2 = list(set(list1) - set(circ_i))
# average color of the image
np.mean(gray_imgs)
# gray out them 
cd OIST/gray
for i in list2:
    sim1 = cv2.circle(sim1, (int(circy[i]+2),int(circx[i]+2)), int(circr[i]+2), color=(38.47,38.47,38.47), thickness=-1)
cv2.imwrite('./shubh2/circ/b_{}.png'.format(j+4), sim1)
'''
# now I want to search for segments 




# convert image into torch tensor for spike data 
import torch
from skimage import img_as_float
im1=[]
for i in range(1):
    #im= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im1.append(np.array(img_as_float(img)))
img1 = np.array([im1])
imd = torch.from_numpy(img1) #[:,:,:,1])
imd.shape

from snntorch import spikegen
spike_data = spikegen.latency(imd, num_steps=70, tau=5, threshold=0.1,clip=True)
import tensorflow as tf 
pil_img = tf.keras.preprocessing.image.array_to_img(np.squeeze(spike_data[1]))
plt.imshow(pil_img)
import snntorch.spikeplot as splt 
num_steps=70
import matplotlib.pyplot as plt
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
#splt.raster(spike_data[:,0].reshape(num_steps,-1)[:,15].unsqueeze(1), ax, s=25, c="black")
splt.raster(np.squeeze(spike_data[1]),ax,c='black')
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
#plt.imshow('gray_imgs)
plt.show()
