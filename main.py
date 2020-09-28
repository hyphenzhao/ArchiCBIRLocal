#!/usr/bin/env python3

"""
This is a simple interface for CBIR system.

Author: Haifeng Zhao
"""
from backend import VGGNet, FileHandler
import os
import numpy as np
import h5py
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename
from fileio import FileIO
from PIL import Image, ImageTk
import sys
from datetime import datetime

# Resolve maximum image size issue
# Add a logger to keep more error information
Image.MAX_IMAGE_PIXELS = None

def GetFoldernameFromSelector(root, e):
	root.update()
	filename = askdirectory()
	if filename != '':
		e.delete(0, END)
		e.insert(0, filename)
	root.update()

def GetFilenameFromSelector(root, e):
	root.update()
	filename = askopenfilename()
	if filename != '':
		e.delete(0, END)
		e.insert(0, filename)
	root.update()

def GenerateFeatureDatabase(fd_name, ft_name, ft_label, pgbar, pglabel, root):
	root.update()
	if os.path.exists(fd_name) and os.path.isdir(fd_name):
		pass
	else:
		messagebox.showerror(title="文件夹读取错误", message="无法读取文件或文件夹不存在")
		return
	ft_label.configure(text=os.path.abspath(ft_name))
	file_handler = FileHandler()
	# img_list = file_handler.get_imlist(fd_name)
	img_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(fd_name) for f in filenames if os.path.splitext(f)[1] == '.jpg']
	# print(img_list)
	print("--------------------------------------------------")
	print("         feature extraction starts")
	print("--------------------------------------------------")
	feats = []
	names = []
	model = VGGNet()
	ite_no = 0
	ite_to = len(img_list)
	if getattr(sys, 'frozen', False):
		application_path = os.path.dirname(sys.executable)
	elif __file__:
		application_path = os.path.dirname(__file__)
	for i, img_path in enumerate(img_list):
		try:
			ite_no += 1
			norm_feat = model.extract_feat(img_path)
			img_name = img_path
			feats.append(norm_feat)
			names.append(img_name.encode('utf-8'))
			print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))
			pgbar['value'] = (100 * ite_no) / ite_to
			print(pgbar['value'])
			label_content="{0:0.2f}%".format(pgbar['value'])
			pglabel.configure(text=label_content)
			root.update()
		except Exception as e:
			logger_fname = os.path.join(self.application_path, 'error.log')
			now = datetime.now()
			dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
			with open(logger_fname, "a+") as f:
				f.write(dt_string + e)
	feats = np.array(feats)
	output = ft_name
	output = os.path.join(application_path, output)
	print("--------------------------------------------------")
	print("      writing feature extraction results ...")
	print("--------------------------------------------------")

	h5f = h5py.File(output, 'w')
	h5f.create_dataset('dataset_1', data = feats)
	h5f.create_dataset('dataset_2', data = np.string_(names))
	h5f.close()

	file_io = FileIO()
	file_io.save_obj(fd_name,"save")


def AnalyseInputImage(queryDir, maxNo, model_name, img_canvas, root):
	h5f = h5py.File(model_name,'r')
	feats = h5f['dataset_1'][:]
	imgNames_utf = h5f['dataset_2'][:]
	imgNames = []
	for i in imgNames_utf:
		imgNames.append(i.decode('utf-8'))
	h5f.close()
	model = VGGNet()
	
	queryVec = model.extract_feat(queryDir)
	scores = np.dot(queryVec, feats.T)
	rank_ID = np.argsort(scores)[::-1]
	rank_score = scores[rank_ID]
	maxres = int(maxNo)
	imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]

	vsbar = Scrollbar(frame_canvas, orient=VERTICAL, command=img_canvas.yview)
	vsbar.grid(row=0, column=1, sticky=NS)
	vsbar.config(command=img_canvas.yview)
	img_canvas.configure(yscrollcommand=vsbar.set)
	frame_images = Frame(img_canvas, bg="grey")
	img_canvas.create_window((0,0), window=frame_images, anchor='nw')
	img_no = 0
	max_in_row = 0
	height_total = 0

	for i in imlist:
		basewidth = 300
		img = Image.open(i)
		wpercent = (basewidth/float(img.size[0]))
		hsize = int((float(img.size[1])*float(wpercent)))
		max_in_row = max(max_in_row, hsize)
		img = img.resize((basewidth,hsize), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(img)
		img_show = Label(frame_images, image=render)
		img_show.image = render
		img_show.grid(row=img_no//3, column=img_no%3)
		img_no += 1
		if img_no%3==0:
			height_total += max_in_row
			max_in_row = 0
	frame_canvas.config(height=height_total)
	root.update()
	img_canvas.config(scrollregion=img_canvas.bbox("all"))
	# root.update()
# Setup main frame
root = Tk()
root.geometry("1000x800")
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=5)
root.columnconfigure(2, weight=5)
root.title("建筑图片聚类系统")

# Setup image canvas
frame_canvas = Frame(root)
frame_canvas.grid(row=6, column=0, columnspan=3)
frame_canvas.grid_rowconfigure(0, weight=1)
frame_canvas.grid_columnconfigure(0, weight=1)
img_canvas = Canvas(frame_canvas, bg="grey", width=900, height=600)
img_canvas.grid(row=0, column=0, sticky="nsew")

# Show feature path
if getattr(sys, 'frozen', False):
	application_path = os.path.dirname(sys.executable)
elif __file__:
	application_path = os.path.dirname(__file__)
feature_name = os.path.join(application_path, "feature.h5")
Label(root, text="提取特征存储文件名称：").grid(row=2, column=0, sticky=W)
if os.path.exists(feature_name):
	feature_text = os.path.abspath(feature_name)
else:
	feature_text = "暂时没有特征被提取"
label_feature_name = Label(root, text=feature_text)
label_feature_name.grid(row=2, column=1, sticky=W)

# Setup progress bar for feature extraction
pgbar = ttk.Progressbar(root, orient=HORIZONTAL, length=550, mode='determinate')
pgbar.grid(row=1, column=1, sticky=W)
l_pg = Label(root, text="0.00%")
l_pg.grid(row=1, column=1, sticky=E)

# If feature file existed, progress bar should be completed status too
if os.path.exists(feature_name):
	pgbar['value'] = 100
	l_pg.configure(text="100.00%")
	root.update()

# Create database and extract features
Label(root, text="图片库路径：").grid(row=0, column=0, sticky=W)
file_io = FileIO()
dbFolder_text = ""
if file_io.check_obj('save'):
	dbFolder_text = file_io.load_obj('save')
e_dbFolder = Entry(root, width=60)
e_dbFolder.grid(row=0, column=1, sticky=W)
e_dbFolder.insert(0, dbFolder_text)
Button(root, text="浏览文件夹", command=lambda: GetFoldernameFromSelector(root, e_dbFolder)).grid(row=0, column=1, sticky=E)
Button(root, text="提取特征", command=lambda: GenerateFeatureDatabase(e_dbFolder.get(), feature_name, label_feature_name, pgbar, l_pg, root)).grid(row=0, column=2, sticky=W)

# Setup the maximum number of similar images
Label(root, text="最大图片显示数量：").grid(row=4, column=0, sticky=W)
e_maxRsltNo = Entry(root, width=5)
e_maxRsltNo.grid(row=4, column=1, sticky=W)
e_maxRsltNo.insert(0, "3")

# Load image to be analysed
Label(root, text="待识别图片：").grid(row=3, column=0, sticky=W)
e_imgInput = Entry(root, width=60)
e_imgInput.grid(row=3, column=1, sticky=W)
Button(root, text="浏览文件", command=lambda: GetFilenameFromSelector(root, e_imgInput)).grid(row=3, column=1, sticky=E)
Button(root, text="识别图片", command=lambda: AnalyseInputImage(e_imgInput.get(), e_maxRsltNo.get(), feature_name, img_canvas, root)).grid(row=4, column=1, sticky=E)

# Output result
Label(root, text="相似图片查找结果：").grid(row=5, column=0, sticky=W)

root.mainloop()