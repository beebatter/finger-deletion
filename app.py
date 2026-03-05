import gradio as gr
import cv2
import sys
import os

# 将项目目录加入系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cascaded_matcher import CascadedMatcher
from src.minutiae_extractor import MinutiaeExtractor
from config import CASCADE

# 全局初始化匹配器
matcher = CascadedMatcher(prefer_deep=True)
# 强制使用90%的判定标准 (与最近的代码改动保持一致)
matcher.cfg['match_decision_threshold'] = 0.90

def process_fingerprints(img1_path, img2_path):
    if not img1_path or not img2_path:
        return None, None, "⚠️ **请上传两张指纹图片进行比对**"
        
    try:
        # 1. 运行级联匹配
        result = matcher.match(img1_path, img2_path, verbose=False)
        
        # 2. 获取预处理后的灰度图和提取的细节点
        _, _, gray_a = matcher._get_skeleton(img1_path)
        _, _, gray_b = matcher._get_skeleton(img2_path)
        
        minutiae_a = matcher._get_minutiae(img1_path)
        minutiae_b = matcher._get_minutiae(img2_path)
        
        # 3. 生成可视化图像（调用库中的 visualize 函数）
        vis_a = MinutiaeExtractor.visualize(gray_a, minutiae_a)
        vis_b = MinutiaeExtractor.visualize(gray_b, minutiae_b)
        
        # Gradio默认显示RGB，OpenCV默认使用BGR，所以我们需要进行颜色空间转换
        vis_a_rgb = cv2.cvtColor(vis_a, cv2.COLOR_BGR2RGB)
        vis_b_rgb = cv2.cvtColor(vis_b, cv2.COLOR_BGR2RGB)
        
        # 4. 构建结果展示文字
        sim_percent = result['final_score'] * 100
        threshold_percent = matcher.cfg['match_decision_threshold'] * 100
        is_match = result['final_score'] >= matcher.cfg['match_decision_threshold']
        
        status = "✅ **匹配成功：系同一指纹**" if is_match else "❌ **匹配失败：为不同指纹**"
        
        md_text = f"### 比对结论：{status}\n\n"
        md_text += f"- **综合相似度**: `{sim_percent:.2f}%` (判定阈值: {threshold_percent:.2f}%)\n"
        md_text += f"- **深度特征相似度**: `{result['deep_score']*100:.2f}%` (AI全局特征比对)\n"
        md_text += f"- **传统细节点相似度**: `{result['minutiae_score']*100:.2f}%` (拓扑特征比对)\n"
        
        if not result['stage1_passed']:
            md_text += "\n⚠️ *注意：在第一阶段深度学习初筛时已被判定为不似，因此相似度极低。*\n"
            
        md_text += f"\n*提取统计：指纹A ({len(minutiae_a)}个特征点) | 指纹B ({len(minutiae_b)}个特征点)*"

        return vis_a_rgb, vis_b_rgb, md_text
        
    except Exception as e:
        return None, None, f"⚠️ **处理过程中发生错误**: {str(e)}"

# 构建Gradio前端界面
with gr.Blocks(title="指纹智能比对系统") as demo:
    gr.Markdown("# 🔍 指纹智能比对系统 (1:1 验证)")
    gr.Markdown("请上传需要比对的两张指纹图像。系统会结合 **ResNet深度特征** 和 **传统细节点(Minutiae)拓扑分布** 进行计算。<br>在下方生成的可视化结果中，**🔴红圈代表端点**，**🔵蓝圈代表分叉点**，线段代表方向。")
    
    with gr.Row():
        with gr.Column():
            img1 = gr.Image(type="filepath", label="上传 第一张 指纹")
            out_img1 = gr.Image(label="第一张指纹 细节点提取可视化")
            
        with gr.Column():
            img2 = gr.Image(type="filepath", label="上传 第二张 指纹")
            out_img2 = gr.Image(label="第二张指纹 细节点提取可视化")
            
    with gr.Row():
        btn = gr.Button("🚀 开始比对", variant="primary", size="lg")
        
    with gr.Row():
        result_display = gr.Markdown("### 等待比对...")
        
    btn.click(fn=process_fingerprints, inputs=[img1, img2], outputs=[out_img1, out_img2, result_display])

if __name__ == "__main__":
    # 绑定0.0.0.0可以在VSCode中方便地通过端口转发在本地浏览器查看
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
