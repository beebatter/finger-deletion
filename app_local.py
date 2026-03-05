import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import sys
import os

# 将项目目录加入系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cascaded_matcher import CascadedMatcher
from src.minutiae_extractor import MinutiaeExtractor

class FingerprintLocalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("指纹智能比对系统 (本地 GUI 版)")
        self.root.geometry("1100x800")
        self.root.configure(bg="#f0f0f0")
        
        # 初始化匹配器
        self.matcher = CascadedMatcher(prefer_deep=True)
        self.matcher.cfg['match_decision_threshold'] = 0.90
        
        self.img1_path = None
        self.img2_path = None
        
        self.create_widgets()

    def create_widgets(self):
        # 顶部标题栏
        title_lbl = tk.Label(self.root, text="🔍 指纹智能比对系统", font=("Arial", 24, "bold"), bg="#f0f0f0")
        title_lbl.pack(pady=10)
        
        desc_lbl = tk.Label(self.root, text="请选择需要比对的两张指纹图像。\n匹配完成后，将在原图展示细节点（🔴端点，🔵分叉点）。", 
                            font=("Arial", 12), bg="#f0f0f0", fg="#555")
        desc_lbl.pack(pady=5)

        # 图像展示区
        self.images_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.images_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # 左侧面板 (指纹 A)
        self.left_panel = tk.Frame(self.images_frame, bg="#fff", bd=2, relief=tk.GROOVE)
        self.left_panel.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)
        
        self.btn1 = tk.Button(self.left_panel, text="📁 选择第一张指纹", font=("Arial", 12), command=self.load_image1, width=20)
        self.btn1.pack(pady=10)
        
        self.lbl_path1 = tk.Label(self.left_panel, text="未选择文件", fg="gray", bg="#fff", wraplength=400)
        self.lbl_path1.pack()
        
        self.canvas1 = tk.Label(self.left_panel, bg="#eaeaea", text="预览图 A", width=40, height=20)
        self.canvas1.pack(pady=10, padx=10, expand=True)

        # 右侧面板 (指纹 B)
        self.right_panel = tk.Frame(self.images_frame, bg="#fff", bd=2, relief=tk.GROOVE)
        self.right_panel.pack(side=tk.RIGHT, padx=20, fill=tk.BOTH, expand=True)
        
        self.btn2 = tk.Button(self.right_panel, text="📁 选择第二张指纹", font=("Arial", 12), command=self.load_image2, width=20)
        self.btn2.pack(pady=10)
        
        self.lbl_path2 = tk.Label(self.right_panel, text="未选择文件", fg="gray", bg="#fff", wraplength=400)
        self.lbl_path2.pack()
        
        self.canvas2 = tk.Label(self.right_panel, bg="#eaeaea", text="预览图 B", width=40, height=20)
        self.canvas2.pack(pady=10, padx=10, expand=True)

        # 控制区
        self.ctrl_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.ctrl_frame.pack(pady=10, fill=tk.X)
        
        self.btn_compare = tk.Button(self.ctrl_frame, text="🚀 开始交叉比对", font=("Arial", 16, "bold"), 
                                     bg="#007bff", fg="white", command=self.compare_images, width=25)
        self.btn_compare.pack(pady=10)

        # 结果输出区
        self.result_text = tk.Text(self.root, height=8, font=("Courier", 12), bg="#282c34", fg="#abb2bf", bd=0, padx=10, pady=10)
        self.result_text.pack(padx=20, pady=10, fill=tk.X)
        self.result_text.insert(tk.END, "等待比对...\n")
        self.result_text.config(state=tk.DISABLED)

    def load_image_generic(self):
        file_path = filedialog.askopenfilename(
            title="选择指纹图片", 
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All Files", "*.*")]
        )
        return file_path

    def load_image1(self):
        path = self.load_image_generic()
        if path:
            self.img1_path = path
            self.lbl_path1.config(text=os.path.basename(path))
            self.display_image(path, self.canvas1)

    def load_image2(self):
        path = self.load_image_generic()
        if path:
            self.img2_path = path
            self.lbl_path2.config(text=os.path.basename(path))
            self.display_image(path, self.canvas2)

    def display_image(self, img_path, label_widget):
        # 使用 OpenCV 读取然后转为 PIL 格式显示
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("错误", f"无法读取图像：{img_path}")
            return
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._update_label_with_cv_img(img_rgb, label_widget)

    def _update_label_with_cv_img(self, cv_img_rgb, label_widget):
        h, w = cv_img_rgb.shape[:2]
        # 保持比例缩放
        max_size = 400
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(cv_img_rgb, (new_w, new_h))
        pil_img = Image.fromarray(resized)
        
        # 保存对 PhotoImage 的引用，防止被垃圾回收
        tk_img = ImageTk.PhotoImage(pil_img)
        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img

    def append_log(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)  # 每次比对前清空
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.config(state=tk.DISABLED)

    def compare_images(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("警告", "请先选择两张需要比对的指纹图片！")
            return
            
        self.btn_compare.config(state=tk.DISABLED, text="比对中...")
        self.root.update()
        
        try:
            # 1. 运行级联匹配
            result = self.matcher.match(self.img1_path, self.img2_path, verbose=False)
            
            # 2. 拿到处理后的特征底图和细节点
            _, _, gray_a = self.matcher._get_skeleton(self.img1_path)
            _, _, gray_b = self.matcher._get_skeleton(self.img2_path)
            
            minutiae_a = self.matcher._get_minutiae(self.img1_path)
            minutiae_b = self.matcher._get_minutiae(self.img2_path)
            
            # 3. 绘制细节点可视化图像
            vis_a = MinutiaeExtractor.visualize(gray_a, minutiae_a)
            vis_b = MinutiaeExtractor.visualize(gray_b, minutiae_b)
            
            # 转为 RGB 以便 Tkinter 显示
            vis_a_rgb = cv2.cvtColor(vis_a, cv2.COLOR_BGR2RGB)
            vis_b_rgb = cv2.cvtColor(vis_b, cv2.COLOR_BGR2RGB)
            
            # 更新界面图
            self._update_label_with_cv_img(vis_a_rgb, self.canvas1)
            self._update_label_with_cv_img(vis_b_rgb, self.canvas2)
            
            # 4. 汇总文本日志
            sim_percent = result['final_score'] * 100
            threshold_percent = self.matcher.cfg['match_decision_threshold'] * 100
            is_match = result['final_score'] >= self.matcher.cfg['match_decision_threshold']
            
            status_symbol = "✅ 匹配成功：系同一指纹" if is_match else "❌ 匹配失败：为不同指纹"
             
            log_text = "=" * 50 + "\n"
            log_text += f" {status_symbol} \n"
            log_text += "=" * 50 + "\n"
            log_text += f"综合相似度:      {sim_percent:.2f}% (系统判定阈值: {threshold_percent:.2f}%)\n"
            log_text += f"深度特征相似度:  {result['deep_score']*100:.2f}%\n"
            log_text += f"拓扑特征相似度:  {result['minutiae_score']*100:.2f}%\n"
            log_text += "-" * 50 + "\n"
            
            if not result['stage1_passed']:
                log_text += "⚠️ 注意：在第一阶段深度学习初筛时已被判定为完全无关，强制失败。\n"
                
            log_text += f"提取统计：A图 ({len(minutiae_a)} 个特征点) | B图 ({len(minutiae_b)} 个特征点)\n"
            
            self.append_log(log_text)
            
        except Exception as e:
            messagebox.showerror("运行错误", str(e))
            self.append_log(f"Error: {e}")
            
        finally:
            self.btn_compare.config(state=tk.NORMAL, text="🚀 开始交叉比对")

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintLocalApp(root)
    root.mainloop()
