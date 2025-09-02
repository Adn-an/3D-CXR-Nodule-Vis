
import sys

import glfw
print(f"Python executable: {sys.executable}")
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from PIL import Image, ImageOps
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(current_dir))
sys.path.append(config_path)
from objloader import * 
# from AR_VIS.Filter import Filter
import argparse
import time
import imgui
from imgui.integrations.glfw import GlfwRenderer
import os
import numpy as np
from PIL import Image
import time

class AR_render:
    
    def __init__(self,  
                 object_paths, 
                 front_path, 
                 side_path,
                 model_scale = 0.03,
                 ct_array=None,
                 ):
        """[Initialize]
        
        Arguments:
            object_path {[string]} -- [your model path]
            model_scale {[float]} -- [your model scale size]
        """
        self.image_w, self.image_h = 1280,720
        self.initOpengl(self.image_w, self.image_h)
        self.model_scale = model_scale
    
        self.object_paths = object_paths
        self.current_model_index = 0
        self.loadModel(self.object_paths[self.current_model_index])
        
        # Model translate that you can adjust by key board 'w', 's', 'a', 'd'
        self.translate_x, self.translate_y, self.translate_z = 0.5, -1, 0
        self.x_rotate, self.y_rotate, self.z_rotate = -70, 0, 200
        self.pre_extrinsicMatrix = None
        
        # self.filter = Filter()
        self.ct_voxels = self.calc_ct(ct_array)
        
        self.show_masks = False
        self.overwrite_masks = False
        self.update_masks = False
        self.recalc_masks = False
        self.do_draw_candidates = True
        self.do_draw_intersections = True
        self.do_draw_ct = False
        self.intersection_voxels = np.empty(0)
        
        # Slider values
        self.slider_width = 50
        self.dragging = ""
        self.hue_low_thresh = 50.0
        self.HUEL = "HUEL"
        self.hue_high_thresh = 150.0
        self.HUEH = "HUEH"
        self.dist_thresh = 0.3
        self.DIST = "DIST"
        
        self.front_path = front_path
        self.side_path = side_path
        
        self.front_texture_id = self.load_texture(self.front_path, mirror = True)
        self.side_texture_id = self.load_texture(self.side_path, mirror = True)
        self.black_texture_id = self.load_texture(f"{current_dir}/black_pixel.png")
        self.front_map = None
        self.side_map = None
        self.calc_masks()
        self.front_mask_id = self.load_texture(f"{current_dir}/mask_front.png", mirror = True)
        self.side_mask_id = self.load_texture(f"{current_dir}/mask_side.png", mirror = True)
        self.rescaled_front = None
        self.rescaled_side = None
        self.calc_candidates()
        self.calc_intersections()

    def calc_masks(self):
        self.front_map = cv2.imread(self.front_path, cv2.IMREAD_UNCHANGED)
        # self.front_map = cv2.flip(self.front_map,1)
        self.calc_mask(self.front_map, 'front')
        self.side_map = cv2.imread(self.side_path, cv2.IMREAD_UNCHANGED)
        # self.side_map = cv2.flip(self.side_map,1)
        self.calc_mask(self.side_map, 'side')

    def calc_mask(self, image, orientation='front'):        
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Create the mask by thresholding the image within the specified hue range
        mask = cv2.inRange(hsv_image, np.array([self.hue_low_thresh,0,0]), np.array([self.hue_high_thresh,255,255]))
        cv2.bitwise_not(mask, mask)
        final_mask = cv2.bitwise_and(mask, mask)
        cv2.imwrite(f'{current_dir}/mask_{orientation}.png', final_mask)

    def loadModel(self, object_path):
        """[loadModel from object_path]
        
        Arguments:
            object_path {[string]} -- [path of model]
        """
        self.model = OBJ(object_path, swapyz = True)
    
    def initOpengl(self, width, height, pos_x=500, pos_y=250, window_name='3D Visualisation of Lung Infections'):
        """[Init OpenGL configuration]

        Arguments:
            width {[int]} -- [width of OpenGL viewport]
            height {[int]} -- [height of OpenGL viewport]

        Keyword Arguments:
            pos_x {int} -- [X coordinate of viewport] (default: {500})
            pos_y {int} -- [Y coordinate of viewport] (default: {500})
            window_name {str} -- [Window name] (default: {'3D Visualisation of Lung Infections'})
        """

        # Initialize GLFW window
        if not glfw.init():
            raise Exception("GLFW cannot be initialized")

        self.window = glfw.create_window(width+153, height, window_name, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window cannot be created")

        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set OpenGL Viewport to adjust for the slider space
        glViewport(0, 0, self.image_w+153, self.image_h)  # Make space for the slider

        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.window)
        
        # Set up event handlers
        glfw.set_key_callback(self.window, self.keyboard_listener)
        
    def draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.image_w / self.image_h, 0.1, 100.0)

        # Set up camera/view
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 5,   # Eye position
                0, 0, 0,   # Look at
                0, 1, 0)   # Up vector

        glTranslatef(self.translate_x, self.translate_y, self.translate_z)
        glScalef(self.model_scale, self.model_scale, self.model_scale)
        glRotatef(self.x_rotate, 1, 0, 0)
        glRotatef(self.y_rotate, 0, 1, 0)
        glRotatef(self.z_rotate, 0, 0, 1)

        self.draw_bounding_box()
        if self.do_draw_candidates:
            self.draw_candidates()
        if self.do_draw_intersections:
            self.draw_intersections(self.intersection_voxels,red=0.0,green=1.0,blue=0.0)
        if self.do_draw_ct:
            self.draw_ct()
        if self.overwrite_masks:
            self.draw_textured_faces(self.black_texture_id,self.black_texture_id)  
        elif self.show_masks:
            self.draw_textured_faces(self.front_mask_id, self.side_mask_id)
        else:
            self.draw_textured_faces(self.front_texture_id, self.side_texture_id)


        # self.draw_textured_faces(self.front_texture_id, self.side_texture_id)

        if self.recalc_masks:
            self.calc_masks()
            self.calc_candidates()
            self.calc_intersections()
            self.update_masks = True
            self.recalc_masks = False
            
        if self.update_masks: # Load textures whenever they change
            # Free old textures
            glDeleteTextures([self.front_mask_id, self.side_mask_id])
            self.front_mask_id = self.load_texture(f"{current_dir}/mask_front.png", mirror=True)
            self.side_mask_id = self.load_texture(f"{current_dir}/mask_side.png", mirror = True)
            self.update_masks = False

        glCallList(self.model.gl_list)

        # self.draw_sliders()
        
        # glfw.swap_buffers(self.window)

        
        
    def draw_bounding_box(self):
        """Draws a bounding box around the 3D model."""
        min_x, min_y, min_z = np.min(self.model.vertices, axis=0)
        max_x, max_y, max_z = np.max(self.model.vertices, axis=0)

        # Define the 8 corner points
        corners = [
            [min_x, min_y, min_z], [max_x, min_y, min_z],
            [max_x, max_y, min_z], [min_x, max_y, min_z],
            [min_x, min_y, max_z], [max_x, min_y, max_z],
            [max_x, max_y, max_z], [min_x, max_y, max_z]
        ]

        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]

        glLineWidth(1.0)
        # glColor3f(1, 0, 0)  # Red color
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(corners[vertex])
        glEnd()
       
    def load_texture(self, image_path, mirror = False):
        """Load an image and convert it into an OpenGL texture."""
        img = Image.open(image_path)
        if mirror:
            img = ImageOps.mirror(img)
        img = img.convert('RGBA')
        img_data = img.tobytes("raw", "RGBA", 0, -1)
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glBindTexture(GL_TEXTURE_2D, 0)  # Unbind texture
        
        return texture_id
     
    def draw_textured_faces(self, front_texture_id, side_texture_id):
        """Draw two faces of the bounding box with a texture applied."""
        if front_texture_id == None or side_texture_id == None:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            print("texture id is none")
            front_texture_id = self.create_transparent_texture()
            side_texture_id = self.create_transparent_texture()
        
        min_x, min_y, min_z = np.min(self.model.vertices, axis=0)
        max_x, max_y, max_z = np.max(self.model.vertices, axis=0)
    
        # Choose a face (e.g., front face)
        front_face_vertices = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, min_y, max_z),
            (min_x, min_y, max_z),
        ]
        
        # Choose a face (e.g., front face)
        side_face_vertices = [
            (min_x, min_y, min_z),
            (min_x, max_y, min_z),
            (min_x, max_y, max_z),
            (min_x, min_y, max_z),
        ]
        
        # Enable texture mapping
        glEnable(GL_TEXTURE_2D)
        
        # back texture
        glBindTexture(GL_TEXTURE_2D, front_texture_id)
        # glColor3f(1, 1, 1)  # Ensure full brightness
        glBegin(GL_QUADS)
        tex_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for i in range(4):
            glTexCoord2f(*tex_coords[i])  # Map texture
            glVertex3f(*front_face_vertices[i])  # Place vertex
        glEnd()
        
        # side texture
        glBindTexture(GL_TEXTURE_2D, side_texture_id)
        glBegin(GL_QUADS)
        tex_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for i in range(4):
            glTexCoord2f(*tex_coords[i])  # Map texture
            glVertex3f(*side_face_vertices[i])  # Place vertex
        glEnd()
        
        glDisable(GL_TEXTURE_2D)  # Disable after drawing
        
        
    def create_transparent_texture(self):
        texture_id = glGenTextures(1)
        transparent_pixel = np.array([0, 0, 0, 0], dtype=np.uint8)  # RGBA = (0,0,0,0)
        
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, transparent_pixel)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        return texture_id
    
    def calc_candidates(self):
        print('begin rescaling candidates')
        start_time = time.time()
        # Get all positive mask coordinates
        front_mask = cv2.flip(cv2.imread(f'{current_dir}/mask_front.png', cv2.IMREAD_GRAYSCALE),-1)
        side_mask = cv2.flip(cv2.imread(f'{current_dir}/mask_side.png', cv2.IMREAD_GRAYSCALE),-1)
        non_black_coords_front = np.column_stack(np.where(front_mask > 0))
        non_black_coords_side = np.column_stack(np.where(side_mask > 0))
        
        # Rescale to value in model
        min_width, min_depth, min_height = np.min(self.model.vertices, axis=0)
        max_width, max_depth, max_height = np.max(self.model.vertices, axis=0)
        front_height, front_width = front_mask.shape
        side_height, side_width = side_mask.shape
        
        # Gather half pixes sizes so that the candidates and intersections are in pixel centers
        px_front_height_half = (0.5/front_height)*(max_height-min_height)
        px_front_width_half = (0.5/front_width)*(max_width-min_width)
        px_side_height_half = (0.5/side_height)*(max_height-min_height)
        px_side_width_half = (0.5/side_width)*(max_depth-min_depth)
        
        # Compute
        rescaled_width_front = px_front_width_half + min_width + non_black_coords_front[:,1] / front_width * (max_width-min_width)
        rescaled_height_front = px_front_height_half + min_height + non_black_coords_front[:,0] / front_height * (max_height-min_height)
        rescaled_width_side = px_side_width_half + min_depth + non_black_coords_side[:,1] / side_width * (max_depth-min_depth)
        rescaled_height_side = px_side_height_half + min_height + non_black_coords_side[:,0] / side_height * (max_height-min_height)
        self.rescaled_front = self.truncate_array(np.column_stack((rescaled_width_front, rescaled_height_front)))
        self.rescaled_side = self.truncate_array(np.column_stack((rescaled_width_side, rescaled_height_side)))
        end_time = time.time()
        print(f'End calculating candidates, elapsed time: {end_time - start_time} seconds')
       
    def truncate_array(self, arr, decimals=6):
        factor = 10 ** decimals
        return np.trunc(arr*factor)/factor
        
    def draw_candidates(self):
        min_width, min_depth, _ = np.min(self.model.vertices, axis=0)
        max_width, max_depth, _ = np.max(self.model.vertices, axis=0)
        # Display as red lines
        delta = 0.03 # so that the lines don't intersact with the images
        min_width = min_width+delta
        min_depth = min_depth+delta
        glColor3f(1.0,0.0,0.0)
        glBegin(GL_LINES)
        for pixel in self.rescaled_front:
            glVertex3f(pixel[0],min_depth,pixel[1])
            glVertex3f(pixel[0],max_depth,pixel[1])
        for pixel in self.rescaled_side:
            glVertex3f(min_width,pixel[0],pixel[1])
            glVertex3f(max_width,pixel[0],pixel[1])
        glEnd()
        glColor3f(1.0,1.0,1.0)       
        
    def get_3d_map_granular_front(self,map_width,map_height,map_depth,
                                  confirmed_height_front_set,confirmed_height_side_set,
                                  thresh_w,thresh_h,thresh_d,
                                  min_width,min_height,min_depth,
                                  max_width,max_height,max_depth,
                                  half_voxel_height):
        # Vectorized approach for checking and appending to intersection_voxels
        map_3d = np.full((map_width, map_height, map_depth), None, dtype=object)
        size = 0
        for f_pixel in self.rescaled_front:
            if f_pixel[1] in confirmed_height_front_set:
                for s_pixel in self.rescaled_side:
                    if s_pixel[1] in confirmed_height_side_set:
                        if abs(f_pixel[1] - s_pixel[1]) <= thresh_h:
                            # Prepare the new voxel array (8 voxels)
                            x = min(round((f_pixel[0]-min_width-thresh_w)/(max_width-min_width)*map_width),map_width-1)
                            # Difference with the other function is here F pixel is used for height
                            y = min(round((f_pixel[1]-min_height)/(max_height-min_height)*map_height),map_height-1)
                            z = min(round((s_pixel[0]-min_depth-thresh_d)/(max_depth-min_depth)*map_depth), map_depth-1)
                            voxel = np.array([
                                [f_pixel[0] - thresh_w, s_pixel[0] + thresh_d, f_pixel[1] + half_voxel_height],
                                [f_pixel[0] + thresh_w, s_pixel[0] + thresh_d, f_pixel[1] + half_voxel_height],
                                [f_pixel[0] - thresh_w, s_pixel[0] - thresh_d, f_pixel[1] + half_voxel_height],
                                [f_pixel[0] + thresh_w, s_pixel[0] - thresh_d, f_pixel[1] + half_voxel_height],
                                [f_pixel[0] + thresh_w, s_pixel[0] + thresh_d, f_pixel[1] - half_voxel_height],
                                [f_pixel[0] - thresh_w, s_pixel[0] + thresh_d, f_pixel[1] - half_voxel_height],
                                [f_pixel[0] - thresh_w, s_pixel[0] - thresh_d, f_pixel[1] - half_voxel_height],
                                [f_pixel[0] + thresh_w, s_pixel[0] - thresh_d, f_pixel[1] - half_voxel_height],
                            ])
                            map_3d[x,y,z] = voxel
                            size += 1

        print(f'number of intersections: {size}')
        size = 0
        return map_3d
        
    def get_3d_map_granular_side(self,map_width,map_height,map_depth,
                                  confirmed_height_front_set,confirmed_height_side_set,
                                  thresh_w,thresh_h,thresh_d,
                                  min_width,min_height,min_depth,
                                  max_width,max_height,max_depth,
                                  half_voxel_height):
        # Vectorized approach for checking and appending to intersection_voxels
        map_3d = np.full((map_width, map_height, map_depth), None, dtype=object)
        size = 0
        for f_pixel in self.rescaled_front:
            if f_pixel[1] in confirmed_height_front_set:
                for s_pixel in self.rescaled_side:
                    if s_pixel[1] in confirmed_height_side_set:
                        if abs(f_pixel[1] - s_pixel[1]) <= thresh_h:
                            # Prepare the new voxel array (8 voxels)
                            x = min(round((f_pixel[0]-min_width-thresh_w)/(max_width-min_width)*map_width),map_width-1)
                            # Difference with the other function is here S pixel is used for height
                            y = min(round((s_pixel[1]-min_height)/(max_height-min_height)*map_height),map_height-1)
                            z = min(round((s_pixel[0]-min_depth-thresh_d)/(max_depth-min_depth)*map_depth), map_depth-1)
                            voxel = np.array([
                                [f_pixel[0] - thresh_w, s_pixel[0] + thresh_d, f_pixel[1] + half_voxel_height],
                                [f_pixel[0] + thresh_w, s_pixel[0] + thresh_d, f_pixel[1] + half_voxel_height],
                                [f_pixel[0] - thresh_w, s_pixel[0] - thresh_d, f_pixel[1] + half_voxel_height],
                                [f_pixel[0] + thresh_w, s_pixel[0] - thresh_d, f_pixel[1] + half_voxel_height],
                                [f_pixel[0] + thresh_w, s_pixel[0] + thresh_d, f_pixel[1] - half_voxel_height],
                                [f_pixel[0] - thresh_w, s_pixel[0] + thresh_d, f_pixel[1] - half_voxel_height],
                                [f_pixel[0] - thresh_w, s_pixel[0] - thresh_d, f_pixel[1] - half_voxel_height],
                                [f_pixel[0] + thresh_w, s_pixel[0] - thresh_d, f_pixel[1] - half_voxel_height],
                            ])
                            map_3d[x,y,z] = voxel
                            size += 1

        print(f'number of intersections: {size}')
        return map_3d
    
    def calc_intersections(self):
        print('begin calculating intersections')
        start_time = time.time()
        
        
        map_width = self.front_map.shape[1]
        map_depth = self.side_map.shape[1]
        map_height = self.front_map.shape[0]
            
        min_width, min_depth, min_height = np.min(self.model.vertices, axis=0)
        max_width, max_depth, max_height = np.max(self.model.vertices, axis=0)
        thresh_h = round(self.dist_thresh * (2/map_height*(max_height-min_height)), 6) #Allow delta to encompass 2 pixels in height
        thresh_w = round((0.5/map_width) * (max_width-min_width), 6)
        thresh_d = round((0.5/map_depth) * (max_depth-min_depth), 6)
        half_voxel_height = round((0.5/map_height) * (max_height-min_height), 6)
        
        unique_height_front = np.unique(self.rescaled_front[:,1])
        unique_height_side = np.unique(self.rescaled_side[:,1])
        
        # Instead of using np.append, collect values in a list and convert to np.array once
        confirmed_height_front = [uhf for uhf in unique_height_front if any(abs(uhf - uhs) <= thresh_h for uhs in unique_height_side)]
        confirmed_height_side = [uhs for uhs in unique_height_side if any(abs(uhs - uhf) <= thresh_h for uhf in unique_height_front)]
        
        # Use set for faster membership checking
        confirmed_height_front_set = set(confirmed_height_front)
        confirmed_height_side_set = set(confirmed_height_side)

        # Pre-allocate for intersection_voxels, an upper bound on the number of voxels
        self.intersection_voxels = np.empty(0)
        # self.intersection_voxels = np.empty((map_width * map_height * map_depth, 8, 3)) # Empty array to hold voxels
        self.intersection_voxels = np.empty((map_width * map_height, 8, 3)) # Empty array to hold voxels

        if self.front_map.shape[0] <= self.side_map.shape[0]:
            map_height = self.side_map.shape[0]
            map_3d = self.get_3d_map_granular_side(map_width,map_height,map_depth,
                                                   confirmed_height_front_set,confirmed_height_side_set,
                                                   thresh_w,thresh_h,thresh_d,
                                                   min_width,min_height,min_depth,
                                                   max_width,max_height,max_depth,
                                                   half_voxel_height)
        else:
            map_3d = self.get_3d_map_granular_front(map_width,map_height,map_depth,
                                                   confirmed_height_front_set,confirmed_height_side_set,
                                                   thresh_w,thresh_h,thresh_d,
                                                   min_width,min_height,min_depth,
                                                   max_width,max_height,max_depth,
                                                   half_voxel_height)
        
        size = 0
        
        for x in [0,map_width-1]:
            for y in range(map_height):
                for z in range(map_depth):
                    if not (map_3d[x,y,z] is None):
                        self.intersection_voxels[size] = map_3d[x,y,z]
                        size += 1
                        
        for x in range(1,map_width-1):
            for y in [0,map_height-1]:
                for z in range(map_depth):
                    if not (map_3d[x,y,z] is None):
                        self.intersection_voxels[size] = map_3d[x,y,z]
                        size += 1
                        
        for x in range(1,map_width-1):
            for y in range(1,map_height-1):
                for z in [0, map_depth-1]:
                    if not (map_3d[x,y,z] is None):
                        self.intersection_voxels[size] = map_3d[x,y,z]
                        size += 1
        
        for x in range(1,map_width-1):
            for y in range(1,map_height-1):
                for z in range(1,map_depth-1):
                    if not (map_3d[x,y,z] is None):
                        if (map_3d[x-1,y,z] is None or map_3d[x+1,y,z] is None or 
                            map_3d[x,y-1,z] is None or map_3d[x,y+1,z] is None or 
                            map_3d[x,y,z-1] is None or map_3d[x,y,z+1] is None):
                            self.intersection_voxels[size] = map_3d[x,y,z]
                            size += 1

        print(f'number of shown voxels {size}')

        self.intersection_voxels = self.intersection_voxels[:size,:,:]
        end_time = time.time()
        print(f'End calculating intersections, elapsed time: {end_time - start_time} seconds') 
    
    def draw_intersections(self, voxels, red=0.0,green=1.0,blue=0.0):
        glColor3f(red,green,blue)
        # Define the faces with references to the unique vertices
        faces = [
            [0, 1, 2, 3],  # Front face
            [4, 5, 6, 7],  # Back face
            [0, 3, 7, 4],  # Right face
            [1, 2, 6, 5],  # Left face
            [0, 1, 5, 4],  # Top face
            [3, 2, 6, 7],  # Bottom face
        ]
        
        # 1. Flatten the voxel vertices into one large array.
        vertices = []
        for voxel in voxels:
            vertices.extend(voxel)

        vertices = np.array(vertices, dtype=np.float32)
        
        # 2. Define the indices for the faces (using the face data)
        indices = []
        for i in range(len(voxels)):
            base_index = i * 8
            for face in faces:
                indices.extend([base_index + vertex for vertex in face])

        indices = np.array(indices, dtype=np.uint32)

        # 3. Create and bind the VBO and IBO
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # 4. Set up vertex pointer (for positions)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_VERTEX_ARRAY)

        # 5. Draw the elements (i.e., the boxes)
        glDrawElements(GL_QUADS, len(indices), GL_UNSIGNED_INT, None)

        # 6. Cleanup
        glDisableClientState(GL_VERTEX_ARRAY)
        
        glColor3f(1.0,1.0,1.0)
        
    def calc_ct(self, ct_array):
        min_width, min_depth, min_height = np.min(self.model.vertices, axis=0)
        max_width, max_depth, max_height = np.max(self.model.vertices, axis=0)
        ct_array = np.array(ct_array)
        height, depth, width = ct_array.shape
        scaled_height = (max_height-min_height)
        scaled_width  = (max_width -min_width)
        scaled_depth  = (max_depth -min_depth)
        px_h_half = (scaled_height/height)/2
        px_w_half = (scaled_width /width) /2
        px_d_half = (scaled_depth /depth) /2
                
        # ct_voxels = np.empty((height * width * depth, 8, 3))
        ct_voxels = np.empty((height * width, 8, 3))
        size = 0
        
        for i in range(height):
            for j in range(depth):
                for k in range(width):
                    if ct_array[i,j,k] == 1:  
                        if (   (not (i+1 < height and ct_array[i+1,j,k] == 1))
                            or (not (j+1 < depth  and ct_array[i,j+1,k] == 1))
                            or (not (k+1 < width  and ct_array[i,j,k+1] == 1))
                            or (not (i-1 >= 0     and ct_array[i-1,j,k] == 1))
                            or (not (j-1 >= 0     and ct_array[i,j-1,k] == 1))
                            or (not (k-1 >= 0     and ct_array[i,j,k-1] == 1))
                            ):
                            x = i/height*scaled_height+min_height
                            y = j/depth *scaled_depth +min_depth
                            z = k/width *scaled_width +min_width
                            voxel = np.array([
                                [z - px_w_half, y + px_d_half, x + px_h_half],
                                [z + px_w_half, y + px_d_half, x + px_h_half],
                                [z - px_w_half, y - px_d_half, x + px_h_half],
                                [z + px_w_half, y - px_d_half, x + px_h_half],
                                [z + px_w_half, y + px_d_half, x - px_h_half],
                                [z - px_w_half, y + px_d_half, x - px_h_half],
                                [z - px_w_half, y - px_d_half, x - px_h_half],
                                [z + px_w_half, y - px_d_half, x - px_h_half],
                            ])
                            ct_voxels[size] = voxel
                            size += 1

        ct_voxels = ct_voxels[:size,:,:]
        print(f"{size} voxels to draw for CT")
                            
        return ct_voxels
        
    def draw_ct(self):
        self.draw_intersections(voxels=self.ct_voxels,red=1.0,green=0.0, blue=1.0)

    def keyboard_listener(self, window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            key_str = glfw.get_key_name(key, scancode)
            print(f"Key pressed: {key_str}") 
            if key_str == '=':
                self.model_scale += 0.01
            elif key_str == '-':
                self.model_scale -= 0.1
            elif key_str == 'w':
                self.translate_x -= 0.1
            elif key_str == 's':
                self.translate_x += 0.1
            elif key_str == 'a':
                self.translate_y -= 0.1
            elif key_str == 'd':
                self.translate_y += 0.1
            elif key_str == 'm':
                self.show_masks = not self.show_masks
                self.overwrite_masks = False
            elif key_str == 'o':
                self.overwrite_masks = not self.overwrite_masks
            elif key_str == 'u':
                self.update_masks = True
            elif key_str == 'x':
                self.x_rotate = (self.x_rotate+5)%360
            elif key_str == 'y':
                self.y_rotate = (self.y_rotate+5)%360
            elif key_str == 'z':
                self.z_rotate = (self.z_rotate+5)%360
            elif key_str == 'c':
                self.do_draw_candidates = not self.do_draw_candidates
            elif key_str == 't':
                self.do_draw_ct = not self.do_draw_ct
            elif key_str == 'i':
                self.do_draw_intersections = not self.do_draw_intersections
            elif key_str == 'r':
                self.model_scale = 0.03
                self.translate_x, self.translate_y, self.translate_z = 0, 0, 0
                self.x_rotate, self.y_rotate, self.z_rotate = 0, 0, 0
        
    def draw_sliders(self):
        imgui.set_next_window_size(400, 100, condition=imgui.ALWAYS)
        imgui.begin("threshold sliders")
        
        imgui.set_next_item_width(300)
        changed, self.hue_low_thresh = imgui.slider_float("Hue Low", self.hue_low_thresh, 0.0, 255.0)
        imgui.set_next_item_width(300)
        changed2, self.hue_high_thresh = imgui.slider_float("Hue High", self.hue_high_thresh, 0.0, 255.0)
        imgui.set_next_item_width(300)
        changed3, self.dist_thresh = imgui.slider_float("Distance", self.dist_thresh, 0.0, 2.0)

        if changed or changed2 or changed3:
            print('set recalc mask')
            self.recalc_masks = True

        imgui.end()
        
    def run(self):
        # Render loop with GLFW
        
        while not glfw.window_should_close(self.window):
            # ms_start = int(round(time.time() * 1000))    
            glfw.poll_events()
            self.imgui_impl.process_inputs()
            imgui.new_frame()
            
            self.draw_scene()
            self.draw_sliders()
            
            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
            # ms_end = int(round(time.time() * 1000))
            # print(f'frame time: {ms_end-ms_start}')
            
            # # Check for mouse button events
            # if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            #     self.mouse_btn_listener(glfw.PRESS)
            # elif glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE:
            #     self.mouse_btn_listener(glfw.RELEASE)
                
        
        self.cleanup_opengl()
        glfw.terminate()
    
    def cleanup_opengl(self):
        """ Free OpenGL resources to prevent memory leaks """
        glDeleteTextures([self.front_texture_id, self.side_texture_id, self.front_mask_id, self.side_mask_id])
        
        # Delete models if they use OpenGL resources
        if hasattr(self, "model") and hasattr(self.model, "gl_list"):
            glDeleteLists(self.model.gl_list, 1)

        print("OpenGL resources cleaned up.")

    def __del__(self):
        """ Destructor to ensure proper cleanup """
        self.cleanup_opengl()
        glfw.terminate()
  

def read_image_count(file_path):
    """Reads the image count from the txt file"""
    with open(file_path, 'r') as f:
        return int(f.read().strip())

def load_images(image_count, sum_nodules_folder, image_size):
    """Load images and create a 3D array of shape (image_count, height, width)"""
    # Create a 3D numpy array to hold the image data
    images_array = np.zeros((image_count, image_size[1], image_size[0]), dtype=int)
    
    print(f'images array shape : {images_array.shape}')

    for i in range(image_count):
        image_file = os.path.join(sum_nodules_folder, f"{i}.png")
        if os.path.exists(image_file):
            # Open the image and convert it to grayscale (0 or 255)
            img = Image.open(image_file).convert('1')  # '1' mode is 1-bit pixels (black and white)
            # Convert image to numpy array (0 for black, 1 for white)
            img_array = np.array(img, dtype=int)
            images_array[i] = img_array
        else:
            # No image found for this slice, so the array stays as 0s (default)
            continue

    return images_array

def load_ct_array():
    parent_folder = "./LIDC-IDRI-0001/ct"
    txt_file_path = os.path.join(parent_folder, "image_count.txt")
    sum_nodules_folder = os.path.join(parent_folder, "sum_nodules")

    # Step 1: Read the image count from the txt file
    image_count = read_image_count(txt_file_path)

    # Step 2: Check the size of the images by loading one of the images
    i = 0
    first_image_path = os.path.join(sum_nodules_folder, f"{i}.png")
    while not os.path.exists(first_image_path) and i < image_count:
        i = i+1
        first_image_path = os.path.join(sum_nodules_folder, f"{i}.png")
        
    if os.path.exists(first_image_path):
        first_image = Image.open(first_image_path)
        image_size = first_image.size  # (width, height)
    else:
        print("Error: No image found!")
        return

    # Step 3: Load all images and create the 3D array
    images_3d_array = load_images(image_count, sum_nodules_folder, image_size)
    
    images_3d_array = images_3d_array[::-1,::-1,::-1]
    
    return images_3d_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-front', 
                        type=str, 
                        # default = f"{current_dir}/overlays/gradcam_map_front_224.png",
                        default = f"{current_dir}/overlays/gradcam_overlay_front_224.png",
                        help="Path to gradcam like image representing front view of lungs")
    parser.add_argument('-side', 
                        type=str, 
                        # default=f"{current_dir}/overlays/gradcam_map_lat_224.png",
                        default=f"{current_dir}/overlays/gradcam_overlay_lat_224.png",
                        help="Path to gradcam like image representing side view of lungs")
    args = parser.parse_args()
    
    front_path = args.front
    side_path = args.side
    
    ct_array = load_ct_array()
    
    model_paths = [f'{config_path}/Models/LungsCombined/Lungs.obj', f'{config_path}/Models/LungsCombined/exported_infection.obj']
    ar_instance = AR_render(model_paths, front_path, side_path, model_scale=0.85, ct_array=ct_array)
    ar_instance.run()