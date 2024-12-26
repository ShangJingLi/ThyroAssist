import gradio as gr
import cv2
import numpy as np
from src.cell_segmentation.image_utils import segmentation_by_threshold, std_cleaner


def process_image(image, input_rl, input_rh, input_gl, input_gh, input_bl, input_bh, input_hl, input_hh,
                  input_sl, input_sh, input_vl, input_vh):
    image = std_cleaner(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    r, g, b = cv2.split(image)
    h, s, v = cv2.split(hsv_image)


    r_seg = segmentation_by_threshold(r, int(input_rl), int(input_rh)).astype(np.bool_)
    g_seg = segmentation_by_threshold(g, int(input_gl), int(input_gh)).astype(np.bool_)
    b_seg = segmentation_by_threshold(b, int(input_bl), int(input_bh)).astype(np.bool_)
    h_seg = segmentation_by_threshold(h, int(input_hl), int(input_hh)).astype(np.bool_)
    s_seg = segmentation_by_threshold(s, int(input_sl), int(input_sh)).astype(np.bool_)
    v_seg = segmentation_by_threshold(v, int(input_vl), int(input_vh)).astype(np.bool_)

    thresh = h_seg & s_seg & v_seg & b_seg & g_seg & r_seg

    # 返回各通道的原图，分割图和thresh图，顺序要与outputs一致
    return (r, r_seg.astype(np.uint8) * 255, g, g_seg.astype(np.uint8) * 255, b, b_seg.astype(np.uint8) * 255,
            h, h_seg.astype(np.uint8) * 255, s, s_seg.astype(np.uint8) * 255, v, v_seg.astype(np.uint8) * 255,
            thresh.astype(np.uint8) * 255)


with gr.Blocks() as demo:
    gr.Markdown("<div style='text-align: center;'><h1>图像分割与阈值处理工具</h1></div>")
    gr.Markdown("### 上传病理图片后，用户可通过手动调节获取每个通道的最优分割阈值，再通过逻辑和运算取得最终分割结果")
    with gr.Row():
        # 图片输入组件
        image_input = gr.Image(label="原始病理图片")

    with gr.Row():
        # 这行input
        rl = gr.Textbox(label="R通道下阈值", value='90')
        rh = gr.Textbox(label="R通道上阈值", value='210')
        gl = gr.Textbox(label="G通道下阈值", value='50')
        gh = gr.Textbox(label="G通道上阈值", value='110')

    with gr.Row():  # 这行output
        r_output = gr.Image(label="R通道原图", elem_id="r", show_label=True, type="numpy")
        r_seg_output = gr.Image(label="R通道分割图", elem_id="r_seg", show_label=True, type="numpy")
        g_output = gr.Image(label="G通道原图", elem_id="g", show_label=True, type="numpy")
        g_seg_output = gr.Image(label="G通道分割图", elem_id="g_seg", show_label=True, type="numpy")

    with gr.Row():
        # 这行input
        bl = gr.Textbox(label="B通道下阈值", value='65')
        bh = gr.Textbox(label="B通道上阈值", value='210')
        hl = gr.Textbox(label="H通道下阈值", value='120')
        hh = gr.Textbox(label="H通道上阈值", value='160')

    with gr.Row():  # 这行output
        b_output = gr.Image(label="B通道原图", elem_id="b", show_label=True, type="numpy")
        b_seg_output = gr.Image(label="B通道分割图", elem_id="b_seg", show_label=True, type="numpy")
        h_output = gr.Image(label="H通道原图", elem_id="h", show_label=True, type="numpy")
        h_seg_output = gr.Image(label="H通道分割图", elem_id="h_seg", show_label=True, type="numpy")


    with gr.Row():
        # 这行input
        sl = gr.Textbox(label="S通道下阈值", value='75')
        sh = gr.Textbox(label="S通道上阈值", value='190')
        vl = gr.Textbox(label="V通道下阈值", value='65')
        vh = gr.Textbox(label="V通道上阈值", value='200')

    with gr.Row():  # 这行output
        s_output = gr.Image(label="S通道原图", elem_id="s", show_label=True, type="numpy")
        s_seg_output = gr.Image(label="S通道分割图", elem_id="s_seg", show_label=True, type="numpy")
        v_output = gr.Image(label="V通道原图", elem_id="v", show_label=True, type="numpy")
        v_seg_output = gr.Image(label="V通道分割图", elem_id="v_seg", show_label=True, type="numpy")

    with gr.Row():
        # 大图窗显示 thresh 图
        thresh_output = gr.Image(label="分割阈值图", elem_id="thresh", show_label=True, type="numpy")

    # 按钮触发处理逻辑
    button = gr.Button("启动分割")
    button.click(process_image,
                 inputs=[image_input, rl, rh, gl, gh, bl, bh, hl, hh, sl, sh,
                        vl, vh],
                 outputs=[r_output, r_seg_output, g_output,g_seg_output, b_output, b_seg_output,
                          h_output, h_seg_output, s_output, s_seg_output, v_output, v_seg_output, thresh_output],
                 )

demo.launch()
