import json
import os
import cv2
import base64
import numpy as np

from inference_onnx import EmotionPredictor

inference=EmotionPredictor("./models/trained_model.onnx")

def lambda_handler(event, context):
    """ AWS lambda handler for inference
        Expected event structure:
        {
            "image_path": "path/to/image.jpg", # optional
            "video_path": "path/to/video.mp4", # optional
            "camera_idx": 0, # optional
            "image_base64": "base64_encoded_image", # optional for direct image data
        }
    """
    try:
        # get input from event
        image_path=event.get('image_path')
        video_path=event.get('video_path')
        camera_idx=event.get('camera_idx')
        
        # image inference
        if image_path:
            if not os.path.exists(image_path):
                return {
                    'statusCode': 404,
                    'body': json.dumps({'error': 'Image not found'})
                }
            # read and process image
            image=cv2.imread(image_path)
            image_result=inference.inference_image(image)
            image_result_path=image_path[:-4]+"_out.png"
            cv2.imwrite(image_result_path,image_result)
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'image_result_path': image_result_path
                })
            }

        # video inference
        elif video_path:
            if not os.path.exists(video_path):
                return {
                    'statusCode': 404,
                    'body': json.dumps({'error': 'Video not found'})
                }
            # inference
            inference.inference_video(video_path)
            video_result_path=video_path[:-4]+"_out.mp4"
            return {
                'statusCode':200,
                'video_result_path': video_result_path
            }
        
        # webcam inference
        elif camera_idx:
            if not isinstance(camera_idx,int) or camera_idx<0:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Invalid camera index'})
                }
            inference.inference_web(camera_idx)
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Webcam inference started'
                })
            }

        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No valid input provided'})
            }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }