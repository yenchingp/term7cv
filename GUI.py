import os
import platform
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

        self.step1_complete = False

        self.detected_objects = tk.StringVar(value='---')
        self.inventory_count = None
        self.cluster_tree = None
        self.selected_cluster = tk.StringVar(value='')

        self.obj_output_path = tk.StringVar(value='')

        self.status = tk.StringVar(value='Idle')
        self.progress_bar = None
        self.silhouette_score = tk.StringVar(value='-')

        self.run_detection_button = None
        self.run_clustering_button = None

        ######### UI Elements Placement #########
        self.frame_1 = self.create_frame_1(root)
        self.frame_1.grid(row=0, column=0, padx=20, pady=10, sticky=tk.N)

        self.frame_2 = self.create_frame_2(root)
        self.frame_2.grid(row=1, column=0, padx=20, pady=10, sticky=tk.N)

        self.status_frame = self.create_status_frame(root)
        self.status_frame.grid(row=2, column=0, padx=20, pady=10, sticky=tk.N)


    ######### UI Element Creation #########
    def create_frame_1(self, container):
        frame_1 = ttk.LabelFrame(container, padding=5, text='Step 1: Object Detection')

        choose_file_button = ttk.Button(frame_1, text='Choose Image...', command=self.choose_image).grid(row=0, column=0, sticky=tk.W)
        file_name_label = ttk.Label(frame_1, textvariable=self.image_name).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        input_image_frame = ttk.Frame(frame_1, width=315, height=315, borderwidth=5, relief='groove')
        input_image_frame.grid(row=1, column=0, columnspan=2, pady=5)
        self.input_image_label = ttk.Label(input_image_frame, image=None)
        self.input_image_label.place(relx=.5, rely=.5, anchor='center')

        self.run_detection_button = ttk.Button(frame_1, text='Run --->', command=self.detection_button_click, state='disabled')
        self.run_detection_button.grid(row=1, column=2, padx=10)

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

        self.run_clustering_button = ttk.Button(frame_2, text='Run Clustering --->', command=self.clustering_button_click, state='disabled')
        self.run_clustering_button.grid(row=1, column=0)

        self.cluster_tree = self.create_cluster_tree(frame_2)
        self.cluster_tree.grid(row=0, column=1, padx=15, rowspan=4)
        self.cluster_tree.bind('<ButtonRelease-1>', self.select_item)
        self.cluster_tree.bind('<Double-1>', self.open_folder)

        cluster_image_frame = ttk.Frame(frame_2, width=315, height=315, borderwidth=5, relief='groove')
        cluster_image_frame.grid(row=0, column=2, rowspan=4, pady=5)
        self.cluster_image_label = ttk.Label(cluster_image_frame, image=None)
        self.cluster_image_label.place(relx=.5, rely=.5, anchor='center')

        return frame_2
    

    def create_status_frame(self, container):
        status_frame = ttk.Labelframe(container, padding=5, text='Status')

        status_label = ttk.Label(status_frame, textvariable=self.status).grid(row=0, column=0, sticky=tk.W)

        self.progress_bar = ttk.Progressbar(status_frame, orient='horizontal', mode='determinate', length=720, maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=5, pady=5)

        return status_frame
    

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

            if self.output_path.get() != '':
                self.run_detection_button['state'] = 'normal'
                self.status.set('Ready to run Object Detection!')


    def choose_output(self):
        output_path = filedialog.askdirectory(title='Choose Output Folder', initialdir='./GUI_image_test')

        if len(output_path) != 0:
            self.output_folder.set(f'/{output_path.split("/")[-1]}')
            self.output_path.set(output_path)

            if self.image_path.get() != '':
                self.run_detection_button['state'] = 'normal'
                self.status.set('Ready to run Object Detection!')


    def detection_button_click(self):
        if self.image_path.get() == '':
            messagebox.showerror(title='Error', message='No Image Selected!')
            return
        
        if self.output_path.get() == '':
            messagebox.showerror(title='Error', message='Please Select your Output Folder!')
            return
        
        self.status.set('Running Object Detection...')
        self.progress_bar['value'] = 25
        root.update_idletasks()

        thread_pool = ThreadPoolExecutor(1)
        detect_status = thread_pool.submit(GUI_utils.detect_objects, self.image_path.get(), self.image_name.get(), self.output_path.get())

        if detect_status.result():
            self.status.set('Detection Complete! Running Object Extraction...')
            self.progress_bar['value'] = 50
            root.update_idletasks()

            extract_status = thread_pool.submit(GUI_utils.extract_objects, self.image_path.get(), self.image_name.get(), self.output_path.get())
        
        if extract_status.result():
            self.step1_complete = True
            self.progress_bar['value'] = 75
            root.update_idletasks()

            thread_pool.shutdown()

        self.display_annotated_img()

        self.obj_output_path.set(f'{self.output_path.get()}/{self.image_name.get().split(".")[0]}_objects')
        self.detected_objects.set(str(len(os.listdir(self.obj_output_path.get()))))

        if self.step1_complete:
            self.run_clustering_button['state'] = 'normal'
            self.status.set('Object Extraction Complete! Ready for Clustering')
            self.progress_bar['value'] = 100
            root.update_idletasks()


    def display_annotated_img(self):
        annotated_img_path = f'{self.output_path.get()}/{self.image_name.get().split(".")[0]}_annotated.jpg'

        img = Image.open(annotated_img_path).resize((300, 300), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        self.output_image_label.configure(image=img)
        self.output_image_label.image=img


    def clustering_button_click(self):
        if not self.step1_complete:
            messagebox.showerror(title='Error', message='Run Object Detection first!')
            return

        self.status.set('Running Clustering! Extracting image features...')
        self.progress_bar['value'] = 10
        root.update_idletasks()
        thread_pool = ThreadPoolExecutor(1)

        extract_features = thread_pool.submit(GUI_utils.extract_features_densenet, self.obj_output_path.get())
        features_dict = extract_features.result()
        self.status.set('Running Clustering! Running PCA...')
        self.progress_bar['value'] = 60
        root.update_idletasks()

        reduce_dims = thread_pool.submit(GUI_utils.dim_reduction_umap, features_dict)
        reduced_features = reduce_dims.result()
        self.status.set('Running Clustering! Determining Clusters...')
        self.progress_bar['value'] = 70
        root.update_idletasks()

        cluster_objs = thread_pool.submit(GUI_utils.clustering_MeanShift, reduced_features, self.obj_output_path.get())
        self.inventory_count, s_score = cluster_objs.result()
        self.silhouette_score.set(str(s_score))
        self.status.set('Running Clustering! Generating Thumbnails...')
        self.progress_bar['value'] = 80
        root.update_idletasks()
        
        self.generate_collages(f'{self.obj_output_path.get()}/clusters')
        self.status.set('Running Clustering! Showing results...')
        self.progress_bar['value'] = 90
        root.update_idletasks()

        self.populate_cluster_tree(self.cluster_tree, self.inventory_count)
        self.status.set(f'Complete! Silhouette Score: {self.silhouette_score.get()}')
        self.progress_bar['value'] = 100
        root.update_idletasks()
    

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
            

    def select_item(self, a):
        tree_item = self.cluster_tree.focus()
        self.selected_cluster.set(self.cluster_tree.item(tree_item)['values'][0])
        
        thumbnail_name = self.selected_cluster.get().lower().replace(" ", "_")
        thumbnail_path = f'{self.obj_output_path.get()}/clusters/thumbnails/{thumbnail_name}.jpg'

        img = Image.open(thumbnail_path).resize((300, 300), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        self.cluster_image_label.configure(image=img)
        self.cluster_image_label.image=img


    def open_folder(self, a):
        file_path = f'{self.obj_output_path.get()}/clusters/{self.selected_cluster.get().lower().replace(" ", "_")}'
        print(file_path)

        if platform.system() == "Windows":
            import os
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            import subprocess
            subprocess.call(["open", "-R", file_path])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", file_path])


########## Main Function ##########
if __name__ == '__main__':
    root = tk.Tk()
    root.title('Inventory Tracker Demo')
    root.resizable(0, 0)
    GUI(root)

    root.mainloop()