import platform
import GUI_utils
import tkinter as tk
from PIL import Image, ImageTk
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

        ######### UI Elements Placement #########
        self.frame_1 = self.create_frame_1(root)
        self.frame_1.grid(column=0, row=0, padx=20, pady=20, sticky=tk.N)


    ######### UI Element Creation #########
    def create_frame_1(self, container):
        frame_1 = ttk.LabelFrame(container, padding=5, text='Step 1: Object Detection')

        choose_file_button = ttk.Button(frame_1, text='Choose Image...', command=self.choose_image).grid(row=0, column=0, padx=5, sticky=tk.W)
        file_name_label = ttk.Label(frame_1, textvariable=self.image_name).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        input_image_frame = ttk.Frame(frame_1, width=315, height=315, borderwidth=5, relief='groove')
        input_image_frame.grid(row=1, column=0, columnspan=2, pady=5)
        self.input_image_label = ttk.Label(input_image_frame, image=None)
        self.input_image_label.place(relx=.5, rely=.5, anchor='center')

        run_button = ttk.Button(frame_1, text='Run --->', command=self.run_button_click).grid(row=1, column=2, padx=10)

        choose_output_button = ttk.Button(frame_1, text='Choose Output Folder...', command=self.choose_output).grid(row=0, column=3, sticky=tk.W)
        output_folder_label = ttk.Label(frame_1, textvariable=self.output_folder).grid(row=0, column=4, sticky=tk.W)
        
        output_image_frame = ttk.Frame(frame_1, width=315, height=315, borderwidth=5, relief='groove')
        output_image_frame.grid(row=1, column=3, columnspan=2, pady=5)
        self.output_image_label = ttk.Label(output_image_frame, image=None)
        self.output_image_label.place(relx=.5, rely=.5, anchor='center')

        return frame_1
    
    def create_frame_2(self, container):
        frame_2 = ttk.LabelFrame(container, padding=5, text='Step 2: Object Clustering')


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


    def choose_output(self):
        output_path = filedialog.askdirectory(title='Choose Output Folder', initialdir='./GUI_image_test')

        if len(output_path) != 0:
            self.output_folder.set(f'/{output_path.split("/")[-1]}')
            self.output_path.set(output_path)


    def run_button_click(self):
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
            thread_pool.shutdown()

        self.display_output()


    def display_output(self):
        annotated_img_path = f'{self.output_path.get()}/{self.image_name.get().split(".")[0]}_annotated.jpg'

        img = Image.open(annotated_img_path).resize((300, 300), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        self.output_image_label.configure(image=img)
        self.output_image_label.image=img


            

########## Main Function ##########
if __name__ == '__main__':
    root = tk.Tk()
    root.title('Inventory Tracker Demo')
    root.resizable(0, 0)
    GUI(root)

    root.mainloop()