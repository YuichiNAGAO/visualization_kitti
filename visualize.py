import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pdb

colors = {
    'Car': 'green',
    'Tram': 'red',
    'Cyclist': 'blue',
    'Van': 'cyan',
    'Truck': 'orange',
    'Pedestrian': 'yellow',
    'Sitter': 'pink',
    'Misc': 'cyan' 
}

rgb_data = {"white": (255, 255, 255),
            "black": (0, 0, 0),
            "blue": (0, 0, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0),
            "pink": (255, 192, 203),
            "cyan": (0, 255, 255)
           }

axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']

data_str={'calib':'txt',
          'image_2':'png',
          'label_2':'txt',
          'velodyne':'bin'}

def get_path(num):
    path={}
    for k,v in data_str.items():
        filename=num+'.'+v
        path[k]=os.path.join(os.getcwd(),'training',k,filename)
    return path



def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
                data[key] = np.array([float(x) for x in value.split()]).reshape(3,-1)
            except:
                pass
    return data

def read_label_file(filepath):
    objects={}
    with open(filepath) as f:
        l_strip = [s.strip() for s in f.readlines()]
        for l in l_strip:
            obj=l.split(' ')
            objects.update({obj[0]:{'2D':[float(value) for value in obj[4:8]] ,'3D':[ float(value) for value in obj[8:15]]}})

    return objects


points=0.2
points_step = int(1. / points)
point_size = 0.01 * (1. / points)


def draw_point_cloud(velo_frame,ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
        
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """
        ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=velo_frame[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
            draw_bb_pt(ax,axes)
        # User specified limits
        if xlim3d!=None:
            ax.set_xlim3d(xlim3d)
        if ylim3d!=None:
            ax.set_ylim3d(ylim3d)
        if zlim3d!=None:
            ax.set_zlim3d(zlim3d)
        
            
            
def draw_2dbb_im(img): 
    for name, bb in objects.items():
        c1 = int(bb['2D'][0]),int(bb['2D'][1])
        c2 = int(bb['2D'][2]),int(bb['2D'][3])
        cv2.rectangle(img, c1, c2, rgb_data[colors[name]],2)
    return img


def draw_3dbb_im(img):
    cam_to_img=calib['P2']
    
    for name, bb in objects.items():
        corners_cam=create_corner(bb)
        corners_im = cam_to_img.dot (corners_cam)
        corners = (corners_im/corners_im[2])[:2]
        
        combs=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,0],[0,5],[1,4],[2,7],[3,6]]
        for i in combs:
            c1=(int(corners[0][i[0]]),int(corners[1][i[0]]))
            c2=(int(corners[0][i[1]]),int(corners[1][i[1]]))
            cv2.line(img, c1 ,c2,rgb_data[colors[name]], 2)
        
    return img

def draw_bb_pt(ax,axes):
    Tr_velo_to_cam = np.zeros((4,4))
    Tr_velo_to_cam[3,3] = 1
    Tr_velo_to_cam[:3,:4] = calib['Tr_velo_to_cam']
    
    R0_rect = np.zeros ((4,4))
    R0_rect[3,3] = 1
    R0_rect[:3,:3] = calib['R0_rect']
    
    cam_to_velo=np.linalg.inv(R0_rect.dot(Tr_velo_to_cam))
    
    for name, bb in objects.items():
        corners_cam=create_corner(bb)
        corners_velo=cam_to_velo.dot(corners_cam)
        corners=(corners_velo/corners_velo[3])[:3]
        
        combs=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,0],[0,5],[1,4],[2,7],[3,6]]
        for i in range(len(combs)):
            ax.plot(corners[axes][:,combs][0][i],corners[axes][:,combs][1][i], colors[name])
        
    
    
def create_corner(bb):
    w=bb['3D'][0]
    h=bb['3D'][1]
    l=bb['3D'][2]
    x=bb['3D'][3]
    y=bb['3D'][4]
    z=bb['3D'][5]
    ry=bb['3D'][6]
    R= np.array([ [+np.cos(ry), 0, +np.sin(ry)],
             [0, 1,               0],
             [-np.sin(ry), 0, +np.cos(ry)] ] )
    x_corners = np.array([0, l, l, l, l, 0, 0, 0]) # -l/2
    y_corners = np.array([0, 0, h, h, 0, 0, h, h]) # -h
    z_corners = np.array([0, 0, 0, w, w, w, w, 0]) # --w/2
    x_corners += -l/2
    y_corners += -h
    z_corners += -w/2
    
    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([x, y, z]).reshape((3,1))
    
    return np.vstack((corners_3D,np.ones((corners_3D.shape[-1]))))

if __name__=='__main__':
    num='000002'
    path=get_path(num)
    objects=read_label_file(path['label_2'])
    calib=read_calib_file(path['calib'])

    f1 = plt.figure(figsize=(15, 8))

    #image
    img1 = cv2.cvtColor( cv2.imread(path['image_2']), cv2.COLOR_BGR2RGB)
    ax1 = f1.add_subplot(111)
    ax1.imshow(img1)
    ax1.set_title('Left RGB Image (cam2)')
    plt.show()
    #velodyne
    with open (path['velodyne'], "rb") as f:
        scan = np.fromfile(f, dtype=np.float32)
        scan = scan.reshape((-1, 4))

    f2 = plt.figure(figsize=(40, 10))
    ax2 = f2.add_subplot(111, projection='3d')
    draw_point_cloud(scan,ax2, 'Velodyne scan')

    f3, ax3 = plt.subplots(3, 1, figsize=(15, 25))
    draw_point_cloud(
        scan,
        ax3[0], 
        'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right', 
        axes=[0, 2] # X and Z axes
    )

    draw_point_cloud(
        scan,
        ax3[1], 
        'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
        axes=[0, 1] # X and Y axes
    )

    draw_point_cloud(
        scan,
        ax3[2], 
        'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane', 
        axes=[1, 2] # Y and Z axes
    )


    f4, ax4 = plt.subplots(2, 1, figsize=(15, 8))
    img_2dbb=draw_2dbb_im(cv2.imread(path['image_2']))
    img_2dbb = cv2.cvtColor(img_2dbb, cv2.COLOR_BGR2RGB)
    ax4[0].imshow(img_2dbb)

    img_3dbb=draw_3dbb_im(cv2.imread(path['image_2']))
    img_3dbb = cv2.cvtColor(img_3dbb, cv2.COLOR_BGR2RGB)
    ax4[1].imshow(img_3dbb)

    plt.show()





        
        

