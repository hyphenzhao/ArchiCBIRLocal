#!/usr/bin/env python3

"""
This is a simple interface for CBIR system.

Author: Haifeng Zhao
"""
from tkinter import *

root = Tk()

Label(root, text="图片库路径：").grid(row=0, column=0)
Entry(root).grid(row=0, column=1)

Label(root, text="提取特征存储文件名称：").grid(row=1, column=0)
label_feature_name = Label(root, text="暂时没有特征被提取").grid(row=1, column=1)

Label(root, text="待识别图片：").grid(row=2, column=0)
Entry(root).grid(row=2, column=1)

Label(root, text="最大图片数量：").grid(row=3, column=0)
Entry(root).grid(row=3, column=1)

Label(root, text="相似图片查找结果：").grid(row=4, column=0)

root.mainloop()