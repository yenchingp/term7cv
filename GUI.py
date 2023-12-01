import os
import random
import GUI_utils
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor


class GUI(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        ############ Global Variables ###########
        self.image_path = tk.StringVar(value='')
        self.image_name = tk.StringVar(value='')

        self.output_path = tk.StringVar(value='')
        self.output_folder = tk.StringVar(value='')

        self.input_image_label = None
        self.output_image_label = None
        self.cluster_image_label = None

        self.extraction_complete = False

        self.detected_objects = tk.StringVar(value='---')
        self.inventory_count = None
        self.cluster_tree = None
        self.selected_cluster = tk.StringVar(value='')

        self.obj_output_path = tk.StringVar(value='')

        ######### UI Elements Placement #########
        self.frame_1 = self.create_frame_1(root)
        self.frame_1.grid(row=0, column=0, padx=20, pady=10, sticky=tk.N)

        self.frame_2 = self.create_frame_2(root)
        self.frame_2.grid(row=1, column=0, padx=20, pady=10, sticky=tk.W)


    ######### UI Element Creation #########
    def create_frame_1(self, container):
        frame_1 = ttk.LabelFrame(container, padding=5, text='Step 1: Object Detection')

        choose_file_button = ttk.Button(frame_1, text='Choose Image...', command=self.choose_image).grid(row=0, column=0, sticky=tk.W)
        file_name_label = ttk.Label(frame_1, textvariable=self.image_name).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        input_image_frame = ttk.Frame(frame_1, width=315, height=315, borderwidth=5, relief='groove')
        input_image_frame.grid(row=1, column=0, columnspan=2, pady=5)
        self.input_image_label = ttk.Label(input_image_frame, image=None)
        self.input_image_label.place(relx=.5, rely=.5, anchor='center')

        run_detection_button = ttk.Button(frame_1, text='Run --->', command=self.detection_button_click).grid(row=1, column=2, padx=10)

        choose_output_button = ttk.Button(frame_1, text='Choose Output Folder...', command=self.choose_output).grid(row=0, column=3, sticky=tk.W)
        output_folder_label = ttk.Label(frame_1, textvariable=self.output_folder).grid(row=0, column=4, sticky=tk.W)
        
        output_image_frame = ttk.Frame(frame_1, width=315, height=315, borderwidth=5, relief='groove')
        output_image_frame.grid(row=1, column=3, columnspan=2, pady=5)
        self.output_image_label = ttk.Label(output_image_frame, image=None)
        self.output_image_label.place(relx=.5, rely=.5, anchor='center')

        return frame_1
    

    def create_frame_2(self, container):
        frame_2 = ttk.LabelFrame(container, padding=5, text='Step 2: Object Clustering')

        text_frame = ttk.Frame(frame_2)
        text_frame.grid(row=0, column=0, sticky=tk.NW)
        ttk.Label(text_frame, text='No. of Objects Detected: ').grid(row=0, column=0, sticky=tk.W)
        detected_objects_label = ttk.Label(text_frame, textvariable=self.detected_objects, width=3).grid(row=1, column=0)

        run_clustering_button = ttk.Button(frame_2, text='Run Clustering --->', command=self.clustering_button_click).grid(row=1, column=0)

        self.cluster_tree = self.create_cluster_tree(frame_2)
        self.cluster_tree.grid(row=0, column=1, padx=15, rowspan=4)
        self.cluster_tree.bind('<ButtonRelease-1>', self.select_item)

        cluster_image_frame = ttk.Frame(frame_2, width=315, height=315, borderwidth=5, relief='groove')
        cluster_image_frame.grid(row=0, column=2, rowspan=4, pady=5)
        self.cluster_image_label = ttk.Label(cluster_image_frame, image=None)
        self.cluster_image_label.place(relx=.5, rely=.5, anchor='center')

        return frame_2
    

    def create_cluster_tree(self, container):
        tree_cols = ['CLU', 'NUM_OBJ']

        tree=ttk.Treeview(container, columns=tree_cols, show='headings', height=15)
        tree.heading('CLU', text='Clusters')
        tree.heading('NUM_OBJ', text='No. of Objects')
        for col in tree['columns']:
            tree.column(col, anchor='center', stretch='no')
        
        tree.column('CLU', anchor='center', stretch='no', width=120)
        tree.column('NUM_OBJ', anchor='center', stretch='no', width=120)

        return tree

    ######### Button Helper Functions #########
    def select_item(self, a):
        tree_item = self.cluster_tree.focus()
        self.selected_cluster.set(self.cluster_tree.item(tree_item)['values'][0])
        
        thumbnail_name = self.selected_cluster.get().lower().replace(" ", "_")
        thumbnail_path = f'{self.obj_output_path.get()}/clusters/thumbnails/{thumbnail_name}.jpg'

        img = Image.open(thumbnail_path).resize((300, 300), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        self.cluster_image_label.configure(image=img)
        self.cluster_image_label.image=img


    def choose_image(self):
        img_path = filedialog.askopenfilename(title='Choose Image to Process', filetypes=[("Image Files", ".jpg .png")], initialdir='./GUI_image_test')
        
        if len(img_path) != 0:
            img_name = img_path.split('/')[-1]

            self.image_path.set(img_path)
            self.image_name.set(img_name)

            img = Image.open(img_path).resize((300, 300), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)

            self.input_image_label.configure(image=img)
            self.input_image_label.image=img


    def choose_output(self):
        output_path = filedialog.askdirectory(title='Choose Output Folder', initialdir='./GUI_image_test')

        if len(output_path) != 0:
            self.output_folder.set(f'/{output_path.split("/")[-1]}')
            self.output_path.set(output_path)


    def detection_button_click(self):
        if self.image_path.get() == '':
            messagebox.showerror(title='Error', message='No Image Selected!')
            return
        
        if self.output_path.get() == '':
            messagebox.showerror(title='Error', message='Please Select your Output Folder!')
            return
        
        thread_pool = ThreadPoolExecutor(1)
        detect_status = thread_pool.submit(GUI_utils.detect_objects, self.image_path.get(), self.image_name.get(), self.output_path.get())
        if detect_status.result():
            print('Detection Complete!')
            extract_status = thread_pool.submit(GUI_utils.extract_objects, self.image_path.get(), self.image_name.get(), self.output_path.get())
        
        if extract_status.result():
            print('Extraction Complete!')
            self.extraction_complete = True
            thread_pool.shutdown()

        self.display_annotated_img()

        self.obj_output_path.set(f'{self.output_path.get()}/{self.image_name.get().split(".")[0]}_objects')
        self.detected_objects.set(str(len(os.listdir(self.obj_output_path.get()))))


    def display_annotated_img(self):
        annotated_img_path = f'{self.output_path.get()}/{self.image_name.get().split(".")[0]}_annotated.jpg'

        img = Image.open(annotated_img_path).resize((300, 300), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        self.output_image_label.configure(image=img)
        self.output_image_label.image=img


    def clustering_button_click(self):
        if not self.extraction_complete:
            messagebox.showerror(title='Error', message='Run Object Detection first!')
            return

        thread_pool = ThreadPoolExecutor(1)

        extract_features = thread_pool.submit(GUI_utils.extract_vgg19_features, self.obj_output_path.get())
        features_dict = extract_features.result()

        print('Features Extracted!')

        reduce_dims = thread_pool.submit(GUI_utils.dim_reduction_umap, features_dict)
        reduced_features = reduce_dims.result()

        print('Dimensionality Reduction Completed!')

        cluster_objs = thread_pool.submit(GUI_utils.clustering_AP, reduced_features, self.obj_output_path.get())
        self.inventory_count = cluster_objs.result()

        print('Objects Clustered!')
        
        self.generate_collages(f'{self.obj_output_path.get()}/clusters')
        
        print('Thumbnails Generated!')

        self.populate_cluster_tree(self.cluster_tree, self.inventory_count)
    

    def populate_cluster_tree(self, tree: ttk.Treeview, inv_count: dict):
        tree.delete(*tree.get_children())

        for (cluster, num) in sorted(inv_count.items()):
            if cluster == -1:
                tree.insert('', tk.END, values=[f'Unclustered (Noise)', num])
            else:    
                tree.insert('', tk.END, values=[f'Cluster {cluster}', num])
    

    def generate_collages(self, cluster_root_path):
        os.makedirs(f'{cluster_root_path}/thumbnails', exist_ok=True)
        
        for cluster_path in os.listdir(cluster_root_path):
            if cluster_path != 'thumbnails':
                image_paths = [os.path.join(f'{cluster_root_path}/{cluster_path}', f) 
                            for f in os.listdir(f'{cluster_root_path}/{cluster_path}') if f.endswith('.jpg')]
                
                num_images = len(image_paths)
                grid_size = min(int(np.sqrt(num_images)), 4)

                image_array = random.choices(image_paths, k=np.square(grid_size))
                image = self.concat_images(image_array, (300, 300), (grid_size, grid_size))
                image.save(f'{cluster_root_path}/thumbnails/{cluster_path}.jpg', 'JPEG')


    def concat_images(self, image_paths, size, shape=None):
        # Open images and resize them
        width, height = size
        images = map(Image.open, image_paths)
        images = [ImageOps.fit(image, size, Image.LANCZOS) 
                for image in images]
        
        # Create canvas for the final image with total size
        shape = shape if shape else (1, len(images))
        image_size = (width * shape[1], height * shape[0])
        image = Image.new('RGB', image_size)
        
        # Paste images into final image
        for row in range(shape[0]):
            for col in range(shape[1]):
                offset = width * col, height * row
                idx = row * shape[1] + col
                image.paste(images[idx], offset)
        
        return image
            

########## Main Function ##########
if __name__ == '__main__':
    root = tk.Tk()
    root.title('Inventory Tracker Demo')
    root.resizable(0, 0)
    GUI(root)

    root.mainloop()