from ultralytics import YOLO
import boto3

model = YOLO("yolov8n.yaml")

training = model.train(data="data.yaml", epochs=100, imgsz=640, device=0)

metrics = model.val()
map50_95 = metrics.box.map
map50 = metrics.box.map50
map75 = metrics.box.map75
maps = metrics.box.maps


#results = model("")

# Export the model to ONNX format
path = model.export(format="onnx")
#save model to s3 bucket 

bucket_name="spongebobpipeline"
role_arn = 'arn:aws:iam::533267059960:role/aws-s3-access'
session_name = 'kubeflow-pipeline-session'
sts_client = boto3.client('sts')
response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
credentials = response['Credentials']
    
#Configure AWS SDK with temporary credentials
s3_client = boto3.client('s3',
                      aws_access_key_id=credentials['AccessKeyId'],
                      aws_secret_access_key=credentials['SecretAccessKey'],
                      aws_session_token=credentials['SessionToken'])