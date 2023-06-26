import cv2

def extract_frame_f_video(video_filename):
    vidcap = cv2.VideoCapture(video_filename)  # Video capture
    ret = False
    
    while ret==False:
        # read the first video frame
        ret,frame = vidcap.read()
        
    # Return the frame as image
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)