import streamlit as st
import cv2
import os
import time
import subprocess
import pandas as pd
from collections import Counter
import supervision as sv
from ultralytics import YOLO
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.inference.predictor import VehiclePredictor

st.set_page_config(page_title="Vehicle AI Recognition", layout="wide", page_icon="car")
st.title("Hệ thống nhận diện thuộc tính phương tiện")
st.markdown("Hỗ trợ nhận diện: **Loại xe** | **Màu sắc** | **Hãng xe**")

st.sidebar.title("Cài đặt")
st.sidebar.markdown("---")

make_threshold = st.sidebar.slider("Confidence Threshold (Hang xe)", 0.0, 1.0, 0.3, 0.05)
inference_interval = st.sidebar.slider("Inference Interval (Toi uu FPS)", 1, 10, 3, 1)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Tải lên Video (.mp4, .avi)", type=["mp4", "avi"])

@st.cache_resource(show_spinner="Dang nap AI Models vao VRAM...")
def load_models():
    yolo_model = YOLO("trained_models/best_yolo_vehicle.pt")
    predictor  = VehiclePredictor(color_model_path="trained_models/best_vehicle_color_resnet50.pth", make_model_path="trained_models/best_vehicle_make_vn.pth")
    return yolo_model, predictor

def reencode_h264(input_path, output_path):
    try:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", output_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

if uploaded_file is not None:
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success("Upload thanh cong!")

    if st.sidebar.button("Bắt đầu", use_container_width=True, type="primary"):
        st.markdown("### Ket qua Phan tich Truc tiep")
        col_video, col_status = st.columns([3, 1])
        with col_video:
            frame_placeholder = st.empty()
        with col_status:
            st.markdown("#### Trang thai")
            status_text  = st.empty()
            progress_bar = st.progress(0)
            fps_text     = st.empty()
            count_text   = st.empty()

        st.markdown("### Log xe da nhan dien")
        log_placeholder = st.empty()

        try:
            yolo_model, predictor = load_models()
            tracker        = sv.ByteTrack()
            box_annotator  = sv.BoxAnnotator(thickness=3)
            label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.0, text_padding=6, text_position=sv.Position.TOP_CENTER)

            TARGET_CLASSES = [0, 1, 2]
            CUSTOM_MAP     = {0: "Car", 1: "Bus", 2: "Truck"}
            VOTING_WINDOW  = 5

            cap          = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            raw_output = os.path.join(temp_dir, "raw_out.mp4")
            fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
            writer     = cv2.VideoWriter(raw_output, fourcc, fps_video, (width, height))

            frame_count    = 0
            track_history  = {}
            vehicle_log    = {}

            status_text.info("Dang xu ly Video...")
            start_time       = time.time()
            frames_processed = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count     += 1
                frames_processed += 1

                results    = yolo_model(frame, verbose=False, conf=0.3)[0]
                detections = sv.Detections.from_ultralytics(results)
                idx        = [i for i, c in enumerate(detections.class_id) if c in TARGET_CLASSES]
                detections = detections[idx]
                detections = tracker.update_with_detections(detections)

                labels = []
                for i in range(len(detections)):
                    tid   = detections.tracker_id[i]
                    xyxy  = detections.xyxy[i]
                    vtype = CUSTOM_MAP.get(detections.class_id[i], "Unknown")
                    x1,y1,x2,y2 = map(int, xyxy)
                    w, h  = x2-x1, y2-y1

                    if tid not in track_history:
                        track_history[tid] = []

                    if w > 40 and h > 40 and frame_count % inference_interval == 0:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            res      = predictor.predict(crop_rgb, yolo_type=vtype)
                            color    = res["color"]["class"].capitalize()
                            make_cls = res["make"]["class"].capitalize()
                            make     = make_cls if res["make"]["confidence"] >= make_threshold else "Unknown"
                            track_history[tid].append((color, make))
                            if len(track_history[tid]) > VOTING_WINDOW:
                                track_history[tid].pop(0)

                    if track_history[tid]:
                        final_color = Counter(p[0] for p in track_history[tid]).most_common(1)[0][0]
                        final_make  = Counter(p[1] for p in track_history[tid]).most_common(1)[0][0]
                    else:
                        final_color, final_make = "Unknown", "Unknown"

                    vehicle_log[tid] = {"Tracking ID": tid, "Loai xe": vtype, "Mau sac": final_color, "Hang xe": final_make if vtype.lower() == "car" else "---"}
                    label = f"ID {tid} {vtype} {final_color} {final_make}" if vtype.lower() == "car" else f"ID {tid} {vtype} {final_color}"
                    labels.append(label)

                annotated = frame.copy()
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                writer.write(annotated)

                if frame_count % 5 == 0:
                    frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                if frame_count % 15 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        fps_text.metric("FPS", f"{frames_processed/elapsed:.1f}")
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    count_text.metric("Xe da phat hien", len(vehicle_log))
                    if vehicle_log:
                        log_placeholder.dataframe(pd.DataFrame(vehicle_log.values()), use_container_width=True, hide_index=True)
                    start_time       = time.time()
                    frames_processed = 0

            cap.release()
            writer.release()

            status_text.info("Dang chuyen doi sang H.264 de xem tren trinh duyet...")
            h264_output = os.path.join(temp_dir, "h264_ket_qua.mp4")
            ok = reencode_h264(raw_output, h264_output)

            status_text.success("Hoan tat! Xem video ket qua ben duoi")
            progress_bar.progress(1.0)

            st.markdown("---")
            st.markdown("### Video ket qua")
            video_to_show = h264_output if (ok and os.path.exists(h264_output)) else raw_output
            with open(video_to_show, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)
            st.download_button("Tai video ket qua ve may", data=video_bytes, file_name="ket_qua_nhan_dien.mp4", mime="video/mp4")

            st.markdown("### Thong ke tong hop")
            if vehicle_log:
                df_final = pd.DataFrame(vehicle_log.values())
                c1, c2, c3 = st.columns(3)
                c1.metric("Tong xe phat hien", len(df_final))
                c2.metric("Xe o to (Car)",     len(df_final[df_final["Loai xe"] == "Car"]))
                c3.metric("Xe khac",            len(df_final[df_final["Loai xe"] != "Car"]))
                st.dataframe(df_final, use_container_width=True, hide_index=True)

        except Exception as e:
            status_text.error(f"Co loi xay ra: {e}")
            st.exception(e)
