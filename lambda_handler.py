import json
import cv2
import numpy as np
import requests
import boto3
import uuid

from inference_onnx import EmotionPredictor

inference=EmotionPredictor("./models/trained_model.onnx")

# initialize S3 client
s3_client=boto3.client('s3')
S3_BUCKET_NAME="images-data-test"

def lambda_handler(event, context):
    """ AWS lambda handler for inference
        {
            "image_url": "https://drive.google.com/uc?id=1GqISERXvrCKxtwJfMKzIKCVi93JAOeSs"
        }
        Return: image_result_ulr
    """
    try:
        # get input from event
        image_url=event.get("image_url")        
        if not image_url:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image URL provided'})
            }
        # download the image
        response=requests.get(image_url)
        if response.status_code!=200:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Failed to fetch image from url'})
            }
        # decode image from url content
        image_arr=np.frombuffer(response.content,np.uint8)
        image=cv2.imdecode(image_arr,cv2.IMREAD_COLOR)
        if image is None:
            return {
                'statusCode':400,
                'body': json.dumps({'error': 'Invalid image data'})
            }
        # process the image
        image_result=inference.inference_image(image)
        # encode the result image to bytes
        _, buffer=cv2.imencode('.png',image_result)
        image_bytes=buffer.tobytes()

        # generate a unique key for the processed image
        image_key = f"processed_images/{uuid.uuid4()}.png"
        # upload to s3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=image_key,
            Body=image_bytes,
            ContentType='image/png',
            ACL='public-read' # make the image publicly accessible
        )

        # generate public url for the image
        image_result_url=f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{image_key}"
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'image_result_url': image_result_url
            })
        }
    except Exception as e:
        return {
            'statusCode':500,
            'body': json.dumps({'error': str(e)})
        }