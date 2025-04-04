import customtkinter
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import os
import glob
from PIL import Image
import cv2
from math import floor
import pandas as pd



customtkinter.set_appearance_mode("Light")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1200x1000")

        self.current_original_image = Image.open("/Users/filipjerenec/Python Projects/GitHub repos/Crack_measurement/aruca.jpg")
        # Class variables
        self.angle = 0  # Initial angle

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        #UI
        ##TabView
        self.dimtabview = customtkinter.CTkTabview(self)
        self.dimtabview.grid(row=0, column=0, sticky="nsew")
        self.dimtabview.add("2D")
        self.dimtabview.add("3D")

        self.dimtabview.tab("2D").grid_columnconfigure(0,weight=1)
        self.dimtabview.tab("2D").grid_rowconfigure(0,weight=1)

        # self.dimtabview.bind("<TabChanged>", self.on_tab_changed)

        ##Create figure
        self.fig = plt.figure(figsize=(2, 2))
        self.ax = self.fig.add_subplot(1,2,1)
        # self.ax.set_xlim(-2, 2)
        # self.ax.set_ylim(-2, 2)
        self.current_image_index = 0
        self.ax.set_aspect(1)
        self.ax.axis('off')

        self.ax2 = self.fig.add_subplot(1,2,2)
        # self.ax2.set_xlim(-2, 2)
        # self.ax2.set_ylim(-2, 2)
        scatter_marker_size = 10
        scatter_marker_linewidths = 0.5
        self.x_dir_line_artist, = self.ax2.plot([0,0],[0,0],linestyle="--",c="r", linewidth = 0.5)
        self.y_dir_line_artist, = self.ax2.plot([0,0],[0,0], linestyle="--",c="r", linewidth = 0.5)
        self.aruco_corner_pts_artist = self.ax2.scatter([0,0,0,0], [0,0,0,0],s=scatter_marker_size,marker = "x", linewidths=scatter_marker_linewidths)
        self.top_pin_pos_artist = self.ax2.scatter([0,0,0,0], [0,0,0,0],s=scatter_marker_size, linewidths=scatter_marker_linewidths)
        self.bot_pin_pos_artist = self.ax2.scatter([0,0,0,0], [0,0,0,0],s=scatter_marker_size, linewidths=scatter_marker_linewidths)
        self.top_pin_vec_pix_artist, = self.ax2.plot([0,0],[0,0])
        self.bot_pin_vec_pix_artist, = self.ax2.plot([0,0],[0,0])
        self.load_line_artist, = self.ax2.plot([0,0],[0,0])
        self.p1_to_ctip_vec_artist, = self.ax2.plot([0,0],[0,0])

        self.ax2.set_aspect(1)
        self.ax2.axis('off')
        # plt.tight_layout()
        plt.subplots_adjust(left=0.001, right=0.999, top=1, bottom=0.0, wspace=0.01, hspace=0.01)

        ##Add figure to 2d tab
        self.plot_frame_2d = customtkinter.CTkFrame(self.dimtabview.tab("2D"))
        self.plot_frame_2d.grid(row=0, column=0, pady=(20, 10), padx=(10, 10), sticky="nsew")

        self.canvas_2d = FigureCanvasTkAgg(self.fig, master=self.plot_frame_2d)
        self.canvas_2d.get_tk_widget().pack(fill="both", expand=True)

        self.bot_frame2d_2 = customtkinter.CTkFrame(master=self)
        self.bot_frame2d_2.grid(row=2,column=0,sticky="nsew")
        self.bot_frame2d_2.grid_columnconfigure(0,weight=1)

        self.slider_frame2d = customtkinter.CTkFrame(master=self.bot_frame2d_2)
        self.slider_frame2d.grid(row=0,column=0,pady=(0,15),padx=(15,15),sticky="nsew")
        self.slider_frame2d.grid_columnconfigure(0,weight=1)

        self.data_frame2d = customtkinter.CTkFrame(master=self.bot_frame2d_2)
        self.data_frame2d.grid(row=1, column=0,padx = (15,15), pady= (0,15),sticky = "nsew")
        self.data_frame2d.grid_columnconfigure(0,weight=1)


        self.measurements_frame1 = customtkinter.CTkFrame(master=self.data_frame2d)
        self.measurements_frame1.grid(row=1, column=0,padx = (15,15), pady= (15,15),sticky = "nsew")
        self.measurements_frame1.grid_columnconfigure(0,weight=1)


        self.measurements_frame2 = customtkinter.CTkFrame(master=self.data_frame2d)
        self.measurements_frame2.grid(row=1, column=1,padx = (15,15), pady= (15,15),sticky = "nsew")
        self.measurements_frame2.grid_columnconfigure(1,weight=2)


        self.measurements_frame3 = customtkinter.CTkFrame(master=self.data_frame2d)
        self.measurements_frame3.grid(row=1, column=2,padx = (15,15), pady= (15,15),sticky = "nsew")
        self.measurements_frame3.grid_columnconfigure(0,weight=1)


        self.slider1_val = customtkinter.StringVar()
        self.slider2_val = customtkinter.StringVar()

        self.slider1_entry = customtkinter.CTkEntry(master=self.slider_frame2d,width=70,textvariable=self.slider1_val)
        self.slider1_entry.grid(row=0,column=1,padx=(10,5))
        self.slider1_entry.bind('<Return>', self.update)


        self.slider2_entry = customtkinter.CTkEntry(master=self.slider_frame2d,width=70,textvariable=self.slider2_val)
        self.slider2_entry.grid(row=1,column=1,padx=(10,5))


        self.org_image_width = customtkinter.DoubleVar()
        self.org_image_height = customtkinter.DoubleVar()

        self.homog_image_height = customtkinter.DoubleVar()
        self.homog_image_width = customtkinter.DoubleVar()

        self.slider_1 = customtkinter.CTkSlider(master=self.slider_frame2d, from_=0, to=1000,
                                                number_of_steps=1000, command=self.slider1_changed)
        self.slider_1.grid(row=0, column=0, padx=(20, 10), pady=(10, 10),sticky="ew")


        self.slider_2 = customtkinter.CTkSlider(master=self.slider_frame2d, from_=0, to=1000,
                                                number_of_steps=1000, command=self.slider2_changed)
        self.slider_2.grid(row=1, column=0, padx=(20, 10), pady=(10, 10),sticky="ew")

        # Create entry to specify as the folder path
        self.folder_path_entry = customtkinter.CTkEntry(master=self.measurements_frame1, placeholder_text="Folder Path")
        self.folder_path_entry.grid(row=0, column=0, padx=(4, 4), pady=(4, 2),sticky="ew")
        # self.folder_path_entry.grid_columnconfigure(0, weight=1)
        # Create button that loads images
        self.image_load_btn = customtkinter.CTkButton(master=self.measurements_frame1, text="Load Images", command=self.load_images)
        self.image_load_btn.grid(row=2, column=0, padx=(4, 4), pady=(2, 4),sticky="ew")
        ## Create a frame from the save and load dimension setting buttons
        self.dimension_config_frame = customtkinter.CTkFrame(master=self.measurements_frame1)
        self.dimension_config_frame.grid_columnconfigure(0,weight=1)
        self.dimension_config_frame.grid_columnconfigure(1,weight=1)
        self.dimension_config_frame.grid(row=3, column=0,padx=(4, 4), pady=(2, 4),sticky="ew")

        # Create a button for loading a data file that contains the dimensions
        self.load_measurement_settings_btn= customtkinter.CTkButton(master=self.dimension_config_frame, text="Load dim. settings")
        self.load_measurement_settings_btn.grid(row=1, column=0,padx=(0, 2), pady=(0, 0),sticky = "ew")

        self.save_measurement_settings_btn= customtkinter.CTkButton(master=self.dimension_config_frame, text="Save dim. settings")
        self.save_measurement_settings_btn.grid(row=1, column=1,padx=(2, 0), pady=(0, 0),sticky = "ew")



        # Create button that calculates the homographic transformation
        self.calc_homography_btn = customtkinter.CTkButton(master=self.measurements_frame3, text="Calculate Homography", command=self.homography_image)
        self.calc_homography_btn.grid(row=0, column=0, padx=(4, 4), pady=(4, 2),sticky="ew")

        # create sub frame for scrolling buttons
        self.scroll_btn_frame = customtkinter.CTkFrame(master = self.measurements_frame3)
        self.scroll_btn_frame.grid(row=1, column=0,padx = (4,4), pady= (2,4),sticky = "nsew")
        self.scroll_btn_frame.grid_columnconfigure(0,weight=1)

        # Create button to go to next and previous picture
        self.next_image_btn = customtkinter.CTkButton(master = self.scroll_btn_frame, text = "Next", command = self.load_next_img)
        self.next_image_btn.grid(row=0, column=1, padx=(2, 0), pady=(0, 0),sticky="ew")

        self.prev_image_btn = customtkinter.CTkButton(master = self.scroll_btn_frame, text = "Prev", command = self.load_prev_img)
        self.prev_image_btn.grid(row=0, column=0, padx=(0, 2), pady=(0, 0),sticky="ew")

        #Create a button to save crack length to list
        self.save_calculation_btn = customtkinter.CTkButton(master = self.measurements_frame3,text="Save Calculation",command=self.save_calc_to_list)
        self.save_calculation_btn.grid(row=2,column=0,padx=(4,4),pady=(0,0),sticky = "ew")

        #Create a button to save all data to file
        self.save_data_btn = customtkinter.CTkButton(master = self.measurements_frame3,text="Save data to file",command=self.save_data_to_file)
        self.save_data_btn.grid(row=3,column=0,padx=(4,4),pady=(4,0),sticky = "ew")


        ## Create entrys for inputing the physical measurements
        #height data
        self.aruco_height_label = customtkinter.CTkLabel(master=self.measurements_frame2,text="Aruco H [mm]")
        self.aruco_height_label.grid(row=0, column=2,padx=(4, 4), pady=(4, 2),sticky="ew")

        self.aruco_height_entry = customtkinter.CTkEntry(master=self.measurements_frame2,width=80,placeholder_text="300")
        self.aruco_height_entry.grid(row=0, column=3,padx=(1, 4), pady=(4, 2))

        #width data
        self.aruco_width_label = customtkinter.CTkLabel(master=self.measurements_frame2,text="Aruco W [mm]")
        self.aruco_width_label.grid(row=0, column=0,padx=(4, 0), pady=(4, 2),sticky="ew")

        self.aruco_width_entry = customtkinter.CTkEntry(master=self.measurements_frame2,width=80,placeholder_text="300")
        self.aruco_width_entry.grid(row=0, column=1,padx=(4, 4), pady=(4, 2))

        #length from top left point in aruco marker detection to the top pinhole center
        self.top_pinhole_x_from_p1_label = customtkinter.CTkLabel(master=self.measurements_frame2,text="Top pin dX")
        self.top_pinhole_x_from_p1_label.grid(row=1,column=0,padx=(2, 0), pady=(2, 4))

        self.top_pinhole_x_from_p1_entry = customtkinter.CTkEntry(master=self.measurements_frame2,width=80,placeholder_text="0.0")
        self.top_pinhole_x_from_p1_entry.grid(row=1, column=1,padx=(1, 0), pady=(2, 4))

        self.top_pinhole_y_from_p1_label = customtkinter.CTkLabel(master=self.measurements_frame2,text="Top pin dY")
        self.top_pinhole_y_from_p1_label.grid(row=1,column=2,padx=(2, 0), pady=(2, 4))

        self.top_pinhole_y_from_p1_entry = customtkinter.CTkEntry(master=self.measurements_frame2,width = 80,placeholder_text="0.0")
        self.top_pinhole_y_from_p1_entry.grid(row=1, column=3,padx=(1, 4), pady=(2, 4))

        #length from top left point in aruco marker detection to the bot pinhole center
        self.bot_pinhole_x_from_p1_label = customtkinter.CTkLabel(master=self.measurements_frame2,text="Bot pin dX")
        self.bot_pinhole_x_from_p1_label.grid(row=2,column=0,padx=(2, 0), pady=(2, 4))

        self.bot_pinhole_x_from_p1_entry = customtkinter.CTkEntry(master=self.measurements_frame2,width=80,placeholder_text="0.0")
        self.bot_pinhole_x_from_p1_entry.grid(row=2, column=1,padx=(1, 0), pady=(2, 4))

        self.bot_pinhole_y_from_p1_label = customtkinter.CTkLabel(master=self.measurements_frame2,text="Bot pin dY")
        self.bot_pinhole_y_from_p1_label.grid(row=2,column=2,padx=(2, 0), pady=(2, 4))

        self.bot_pinhole_y_from_p1_entry = customtkinter.CTkEntry(master=self.measurements_frame2,width = 80,placeholder_text="0.0")
        self.bot_pinhole_y_from_p1_entry.grid(row=2, column=3,padx=(1, 4), pady=(2, 4))

        #Label and entru for number of cycles
        self.num_of_cycles_label = customtkinter.CTkLabel(master=self.measurements_frame2,text="Cycle num.")
        self.num_of_cycles_label.grid(row=3,column=0, padx=(2, 0), pady=(2, 4))

        self.num_of_cycles_entry = customtkinter.CTkEntry(master=self.measurements_frame2,width=80,placeholder_text="0.0")
        self.num_of_cycles_entry.grid(row=3, column=1,padx=(1, 0), pady=(2, 4))
        # self.num_of_cycles_entry.bind("<Return>", self.num_of_cycles_entry_return_event)

        # self.ctip_x_from_p1_label = customtkinter.CTkLabel(master=self.measurements_frame2,text="Ctip dX")
        # self.ctip_x_from_p1_label.grid(row=3,column=0,padx=(2, 0), pady=(2, 4))

        # self.ctip_x_from_p1_entry = customtkinter.CTkEntry(master = self.measurements_frame2,width=50,placeholder_text="0.0")
        # self.ctip_x_from_p1_entry.grid(row=3,column=1,padx=(1, 0), pady=(2, 4))

        # self.ctip_y_from_p1_label = customtkinter.CTkLabel(master=self.measurements_frame2,text="Ctip dY")
        # self.ctip_y_from_p1_label.grid(row=3,column=2,padx=(2, 0), pady=(2, 4))

        # self.ctip_y_from_p1_entry = customtkinter.CTkEntry(master = self.measurements_frame2,width=50,placeholder_text="0.0")
        # self.ctip_y_from_p1_entry.grid(row=3,column=3,padx=(1, 4), pady=(2, 4))

        # Initialize aruco detection
        # Initialize QR Code detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Adjust detection settings
        self.aruco_params.adaptiveThreshWinSizeMin = 3  # Smaller window size for adaptive thresholding
        self.aruco_params.adaptiveThreshWinSizeMax = 50
        self.aruco_params.adaptiveThreshWinSizeStep = 15
        self.aruco_params.minMarkerPerimeterRate = 0.1  # Minimum size of marker as a fraction of total image
        self.aruco_params.maxMarkerPerimeterRate = 15.0  # Maximum size of marker as a fraction of total image
        self.aruco_params.polygonalApproxAccuracyRate = 0.01  # Lower this if contours are missed
        self.aruco_params.minCornerDistanceRate = 0.01  # Allow corners to be closer
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Phisycal size of the QR code, measure between the top two points
        self.qr_code_size = 5.0 #mm
        # Relative position of the QR code to the load line
        ## X dist defined as the direction from QRTR to Load Line POints -> QRTL = QR code Top Left point, QRTR = QR code Top Right point
        ## When inputing these distances, you need to input the correct signs -> input the actual vector and not just the absolute value
        ## The actual vector is directed from QRTL to a0. So if the vector is directed to the top and to the left of the QRTL both have -!
        self.x_dist_from_QRTL_vhx = -12.658 #mm
        self.y_dist_from_QRTL_vhx = 13.105 #mm
        # Side length of the square onto which the QR code points are homographically transformed
        self.sq_len = 300
        self.aruco_height = customtkinter.DoubleVar
        self.aruco_width = customtkinter.DoubleVar
        self.aruco_height = 500    #pixels
        self.aruco_width  = 500    #pixels

    def num_of_cycles_entry_return_event(self,event):
        self.cycles = float(self.num_of_cycles_entry.get())
        print(self.cycles)
        return
    def load_next_img(self):
        self.first_calculation_for_image = True
        if self.current_image_index == self.num_loaded_images:
            print("Current image index = number of images. There is no next image.")
            return
        self.current_image_index += 1
        self.current_original_image = self.images[self.current_image_index]
        self.ax.imshow(self.current_original_image)
        image_name = os.path.basename(self.image_paths[self.current_image_index])
        self.fig.suptitle(image_name,fontsize=8)
        self.ax.set_title(f"Image index = {self.current_image_index}",fontsize=8)
        self.fig.canvas.draw()

    def load_prev_img(self):
        self.first_calculation_for_image = True
        if self.current_image_index == 0:
            print("Current image index = 0. There is no previous image.")
            return
        self.current_image_index -= 1
        self.current_original_image = self.images[self.current_image_index]
        self.ax.imshow(self.current_original_image)
        image_name = os.path.basename(self.image_paths[self.current_image_index])
        self.ax.set_title(f"Image index = {self.current_image_index}",fontsize=8)
        self.fig.suptitle(image_name,fontsize=8)
        self.fig.canvas.draw()
        return

    def get_image_folder_path(self):
        return self.folder_path_entry.get()

    def save_calc_to_list(self):
        self.results[self.current_image_index][0] = os.path.basename(self.image_paths[self.current_image_index])
        self.results[self.current_image_index][1] = self.current_image_index
        self.results[self.current_image_index][2] = self.crack_length
        self.results[self.current_image_index][3] = float(self.num_of_cycles_entry.get())
        return

    def save_data_to_file(self):


        df = pd.DataFrame(self.results, columns=["Image Name", "Current Image Index","Crack Length","Number of Cycles"])
        df.to_csv("output.txt", sep="\t", index=False)
        print("File saved successfully!")
        return

    def load_images(self):
        self.first_calculation_for_image = True
        # Define the folder path
        self.current_image_index = 0
        self.ax.set_title(f"Image index = {self.current_image_index}",fontsize=8)
        # folder_path = self.get_image_folder_path()
        folder_path = "/Users/filipjerenec/Python Projects/GitHub repos/Crack_measurement/MPBL1_images"

        # Use glob to get all .jpg and .png files (you can add more extensions as needed)
        self.image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
        self.image_paths += glob.glob(os.path.join(folder_path, "*.png"))

        # Load images into a list
        self.images = []
        for path in self.image_paths:
            try:
                img = Image.open(path)
                self.images.append(img)
                print(f"Loaded {path}")
            except Exception as e:
                print(f"Failed to load {path}: {e}")
        self.num_loaded_images = len(self.images)
        # Create an array to save the crack lengths to
        self.results = [[0.0 for _ in range(4)] for _ in range(len(self.images))]


        print(f"Total images loaded: {len(self.images)}")
        self.current_original_image = self.images[0]
        image_name = os.path.basename(self.image_paths[0])
        self.fig.suptitle(image_name,fontsize=8)
        # self.ax.set_title(image_name,fontsize=5)
        self.ax.imshow(self.current_original_image)
        self.fig.canvas.draw()

    def slider1_entry_return_key_event(self,event):
        return

    def get_float(self, entry):
        """Safely converts entry text to float, returns None if invalid."""
        try:
            entry.configure(border_color="gray")
            return float(entry.get())
        except ValueError:
            entry.configure(border_color="red")
            print(f"Entry input error. Did you input a number?")
            return 0.0  # Or a default value like 0.0

    def update(self,event):
        return

    def draw_lines(self):
        current_height = self.current_original_image.height
        current_width = self.current_original_image.width

        current_height_homography = self.homography_img.shape[0]
        current_width_homography = self.homography_img.shape[1]

        value1 = self.slider_1.get()
        value2 = self.slider_2.get()

        self.x_dir_line_artist.set_xdata([value1,value1])
        self.x_dir_line_artist.set_ydata([0,current_height_homography])

        self.y_dir_line_artist.set_xdata([0,current_width_homography])
        self.y_dir_line_artist.set_ydata([value2,value2])

        self.fig.canvas.draw()

    def slider1_changed(self, value):
        self.slider1_val.set(value)
        self.draw_lines()
        return

    def slider2_changed(self, value):
        self.slider2_val.set(value)
        self.draw_lines()
        return

    def calc_crack_length(self):
        top_pin_dx = self.get_float(self.top_pinhole_x_from_p1_entry)
        top_pin_dy = self.get_float(self.top_pinhole_y_from_p1_entry)
        top_pin_vec = np.array([top_pin_dx,top_pin_dy])
        print(f"top_pin_vec {top_pin_vec}")

        bot_pin_dx = self.get_float(self.bot_pinhole_x_from_p1_entry)
        bot_pin_dy = self.get_float(self.bot_pinhole_y_from_p1_entry)
        bot_pin_vec = np.array([bot_pin_dx,bot_pin_dy])
        print(f"bot_pin_vec {bot_pin_vec}")


        self.ctip_x_pos_pixels =float(self.slider_1.get())
        self.ctip_y_pos_pixels =float(self.slider_2.get())
        ####-----------
        self.aruco_pt1_x_pos_pixels = (self.aruco_corner_points[0,0])
        self.aruco_pt1_y_pos_pixels = (self.aruco_corner_points[0,1])
        self.aruco_pt2_x_pos_pixels = (self.aruco_corner_points[1,0])
        self.aruco_pt2_y_pos_pixels = (self.aruco_corner_points[1,1])
        real_distance_per_pixel = self.qr_code_size/np.sqrt((self.aruco_pt2_x_pos_pixels - self.aruco_pt1_x_pos_pixels)**2 + (self.aruco_pt2_y_pos_pixels-self.aruco_pt1_y_pos_pixels)**2) #[mm/pixel]
        real_pixel_per_distance = 1/real_distance_per_pixel
        ####-----------

        self.top_pin_pos_x_pixels = self.aruco_pt1_x_pos_pixels + top_pin_dx*real_pixel_per_distance
        self.top_pin_pos_y_pixels = self.aruco_pt1_y_pos_pixels + top_pin_dy*real_pixel_per_distance

        self.bot_pin_pos_x_pixels = self.aruco_pt1_x_pos_pixels + bot_pin_dx*real_pixel_per_distance
        self.bot_pin_pos_y_pixels = self.aruco_pt1_y_pos_pixels + bot_pin_dy*real_pixel_per_distance


        ctip_vec_pixels = np.array([self.ctip_x_pos_pixels-self.aruco_pt1_x_pos_pixels, self.ctip_y_pos_pixels-self.aruco_pt1_y_pos_pixels])
        ctip_vec = ctip_vec_pixels * real_distance_per_pixel
        rot_matrix= np.array([[0,-1],
                              [1,0]])

        print(f"ctip_vec_pixels = {ctip_vec_pixels}")
        print(f"ctip_vec = {ctip_vec}")

        load_line_normal = (bot_pin_vec - top_pin_vec)/np.linalg.norm(bot_pin_vec - top_pin_vec)
        load_line_binormal = rot_matrix @ load_line_normal
        print(f"load_line_normal = {load_line_normal}")
        print(f"load_line_nbiormal = {load_line_binormal}")
        

        top_pin_to_ctip_vec = ctip_vec - top_pin_vec  
        print(f"top_pin_to_ctip_vec = {top_pin_to_ctip_vec}")

        crack_length = np.dot(top_pin_to_ctip_vec, load_line_binormal)

        return crack_length

    def homography_image(self):

        self.ax2.set_title(f"Image index = {self.current_image_index}",fontsize=8)
        image_np = np.array(self.current_original_image)
        image = cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
        image_gr = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)



        decoded_info, ids, rejected = self.detector.detectMarkers(image_gr)

        # image_with_markers = cv2.aruco.drawDetectedMarkers(image.copy(), decoded_info, ids)
        # cv2.imshow("Detected Markers", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if ids == None:
            print("No marker detected")

        src_pts = decoded_info[0].reshape(4, 2).astype(np.float32)  # Assumes one marker is detected

        # Define the points onto which the QR code points are transformed
        ## In this definition the QR code points get transformed to the coordinate systems origin
        ## The image pixels that get transformed to negative position are lost
        ## To correct this I use this homography to find the outer points of the image in the transformed image
        ## and then use those positions to calculate the translations needed to get the whole image in the cv2 window when transformed
        # dst_pts = (np.array([[0,0],
        #                     [self.sq_len,0],
        #                     [self.sq_len,self.sq_len],
        #                     [0,self.sq_len]], dtype=np.float32))
        dst_pts = (np.array([[0,0],
                            [self.aruco_width,0],
                            [self.aruco_width,self.aruco_height],
                            [0,self.aruco_height]], dtype=np.float32))

        # Calculate the homography matrix
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if homography_matrix is None:
            print("Error: Homography matrix is not valid.")

        src_corner_pts = np.array([ [0, 0],
                                    [image.shape[1], 0],
                                    [image.shape[1], image.shape[0]],
                                    [0, image.shape[0]]],
                                    dtype=np.float32)

        # Apply the homography transformation to the corner positions of the image to get the outer edge positions in the transfomed image.
        warped_corner_pts = cv2.perspectiveTransform(src_corner_pts.reshape(-1, 1, 2), homography_matrix).reshape(-1,2)

        #Create a second homography matrix that includes the proper translation
        minx_translate = np.min(warped_corner_pts[:,0])
        miny_translate = np.min(warped_corner_pts[:,1])

        #Create the translation corrected points onto which the QR code points are transformed
        dst_pts2 = dst_pts - np.array([minx_translate, miny_translate])

        homography_matrix_corrected, _ = cv2.findHomography(src_pts , dst_pts2, cv2.RANSAC, 5.0)
        warped_corner_pts_corrected =   cv2.perspectiveTransform(src_corner_pts.reshape(-1, 1, 2),
                                                                homography_matrix_corrected).reshape(-1,2)
        warped_points_corrected =       cv2.perspectiveTransform(src_pts.reshape(-1,1,2),
                                                                homography_matrix_corrected).reshape(-1,2)
        if homography_matrix_corrected is None:
                print("Error: Homography matrix is not valid.")

        warped_image_corrected = cv2.warpPerspective(image, homography_matrix_corrected,
                                                    (floor(np.max(warped_corner_pts_corrected[:,0])),
                                                    floor(np.max(warped_corner_pts_corrected[:,1]))))




        if warped_image_corrected is None or warped_image_corrected.size == 0:
            print("Error: warped_image_corrected is empty.")

        self.homography_img = cv2.cvtColor(warped_image_corrected,cv2.COLOR_BGR2RGB)

        new_height = self.homography_img.shape[0]
        new_width = self.homography_img.shape[1]
        self.slider_1.configure(from_ = 0, to =new_width, number_of_steps = new_width)
        self.slider_2.configure(from_ = 0, to =new_height, number_of_steps = new_height)

        if self.first_calculation_for_image == True:
            self.ax2.imshow(self.homography_img)
        
        self.first_calculation_for_image = False

        # self.ax2.scatter(warped_points_corrected[:,0], warped_points_corrected[:,1])
        new_points = np.column_stack([warped_points_corrected[:,0], warped_points_corrected[:,1]])
        self.aruco_corner_points = warped_points_corrected

        self.crack_length = self.calc_crack_length()
        new_top_pin_pos = np.column_stack([self.top_pin_pos_x_pixels,self.top_pin_pos_y_pixels])
        new_bot_pin_pos = np.column_stack([self.bot_pin_pos_x_pixels,self.bot_pin_pos_y_pixels])

        self.aruco_corner_pts_artist.set_offsets(new_points)
        self.top_pin_pos_artist.set_offsets(new_top_pin_pos)
        self.bot_pin_pos_artist.set_offsets(new_bot_pin_pos)
        # self.top_pin_vec_pix_artist.set_xdata([self.aruco_pt1_x_pos_pixels,self.top_pin_pos_x_pixels])
        # self.top_pin_vec_pix_artist.set_ydata([self.aruco_pt1_y_pos_pixels,self.top_pin_pos_y_pixels])

        # self.bot_pin_vec_pix_artist.set_xdata([self.aruco_pt1_x_pos_pixels,self.bot_pin_pos_x_pixels])
        # self.bot_pin_vec_pix_artist.set_ydata([self.aruco_pt1_y_pos_pixels,self.bot_pin_pos_y_pixels])

        self.load_line_artist.set_xdata([self.top_pin_pos_x_pixels,self.bot_pin_pos_x_pixels])
        self.load_line_artist.set_ydata([self.top_pin_pos_y_pixels,self.bot_pin_pos_y_pixels])

        self.p1_to_ctip_vec_artist.set_xdata([self.aruco_pt1_x_pos_pixels,self.ctip_x_pos_pixels])
        self.p1_to_ctip_vec_artist.set_ydata([self.aruco_pt1_y_pos_pixels,self.ctip_y_pos_pixels])
        self.fig.canvas.draw()
        return


class Specimen():
    def __init__(self):
        self.specimen = None


app = App()
app.mainloop()