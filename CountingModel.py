import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  #solve the dead kernel problem
HOME = os.getcwd()
# os.chdir(HOME)
# os.chdir(HOME+ '\ByteTrack')
# print(os.getcwd())
# from IPython import display
# import ultralytics
# import yolox
# import ByteTrack.yolox as yolox
# print("yolox.__version__:", yolox.__version__)
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

@dataclass(frozen=True) 
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

import supervision
print("supervision.__version__:", supervision.__version__)
from supervision.draw.color import ColorPalette
# from supervision.geometry.dataclasses import Point
from supervision import VideoInfo
from supervision import get_video_frames_generator
from supervision import VideoSink
# from supervision.notebook.utils import show_frame_in_notebook
from supervision import Detections, BoxAnnotator
# from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from typing import List
import numpy as np

videosFolderURL = "files/modelVideos/"

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

def countingModel(videoname, resultVideoName):
    # settings
    os.chdir(HOME)
    SOURCE_VIDEO_PATH = f"{HOME}/{videosFolderURL}{videoname}"
    TARGET_VIDEO_PATH = f"{HOME}/{videosFolderURL}{resultVideoName}"
    MODEL = "best.pt"
    from ultralytics import YOLO

    model = YOLO(MODEL)
    model.fuse()
    # dict maping class_id to class_name
    CLASS_NAMES_DICT = model.model.names
    # class_ids of interest - car, motorcycle, bus and truck
    CLASS_ID = [0]
    # create frame generator
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
    # create instance of BoxAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    # acquire first video frame
    iterator = iter(generator)
    frame = next(iterator)
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame)
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )

    # format custom labels
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ] 
    # annotate and display frame
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

    VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    from tqdm import tqdm # to show the progress

    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    # create frame generator
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=2)
    unique_tracker_ids = set() #to count the number of apples

    # open target video file
    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        # loop over video frames
        for frame in tqdm(generator, total=video_info.total_frames):
            # model prediction on single frame and conversion to supervision Detections
            results = model(frame)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            
            
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            
            # format custom labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            #line_annotator.annotate(frame=frame, line_counter=line_counter)
            sink.write_frame(frame)
            
        for tracker_id in detections.tracker_id:
            
                unique_tracker_ids.add(tracker_id)
    apples_number = len(unique_tracker_ids)
    # print(f"Number of apples are {apples_number}")
    return apples_number