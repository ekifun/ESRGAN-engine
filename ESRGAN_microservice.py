from flask import Flask, request, jsonify
import os
import glob
import torch
import cv2
import numpy as np
import json
import redis
import RRDBNet_arch as arch
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
PORT = int(os.getenv('PORT', 7001))
# Initialize Redis client
redis_client = redis.Redis(
    host='localhost',   # Redis server hostname (default: 'localhost')
    port=6379,          # Redis server port (default: 6379)
    db=0,               # Database number (default: 0)
    decode_responses=True  # Decode byte responses to strings (helpful for working with strings)
)
PROCESSING_TOPICS_KEY = "processingTopics"
PROCESSED_TOPICS_KEY = "processedTopics"

# Define the pub/sub channel for task completion
PUB_SUB_CHANNEL = 'task_completed'

# Store topic processing statuses
topics = {}

# Model setup (ESRGAN)
model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)
test_img_folder = 'LR/*'

def process_image(image_path, topic_id):
    """Run ESRGAN model on the given image and update the topic status."""
    logging.info(f"process_image image_path: {image_path}, topic_id: {topic_id}")

    # Update topic status to 'processing'
    topics[topic_id] = {"status": "processing", "progress": 0}
    

    # Simulate processing
    time.sleep(2)  # Simulate delay before processing

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    result_path = f'results/{os.path.splitext(os.path.basename(image_path))[0]}_rlt.png'
    cv2.imwrite(result_path, output)
    logging.info(f"process_image result_path: {result_path}")

    # Publish to the PUB_SUB_CHANNEL that the task is completed
    redis_client.publish(PUB_SUB_CHANNEL, json.dumps({"topic_id": topic_id, "status": "processed", "result": result_path}))
    logging.info(f"Published task completion for topic '{topic_id}' to Pub/Sub channel '{PUB_SUB_CHANNEL}'.")

    # Update status to 'processed'
    topics[topic_id] = {"status": "processed", "progress": 100, "result": result_path}

# API to create a new topic
@app.route('/create_topic', methods=['POST'])
def create_topic():
    topic_id = str(len(topics) + 1)
    topics[topic_id] = {"status": "created"}

    # Start processing images asynchronously
    for path in glob.glob(test_img_folder):
        logging.info(f"create_topic path: {path}")
        # This actually starts the thread, making the function process_image execute in parallel for each image.
        threading.Thread(target=process_image, args=(path, topic_id)).start()

    return jsonify({"topic_id": topic_id}), 201 #The 201 HTTP status code means that a resource has been created. This indicates that the server successfully started the processing request.

# API to get the status of a topic
@app.route('/get_topic/<topic_id>', methods=['GET'])
def get_topic(topic_id):
    if topic_id in topics:
        logging.info(f"get_topic topic_id: {topic_id} topics[topic_id]: {topics[topic_id]}")
        return jsonify(topics[topic_id]), 200
    else:
        return jsonify({"error": "Topic not found"}), 404

# API to close a topic (cleanup)
@app.route('/close_topic/<topic_id>', methods=['POST'])
def close_topic(topic_id):
    if topic_id in topics:
        del topics[topic_id]
        return jsonify({"message": "Topic closed successfully"}), 200
    else:
        return jsonify({"error": "Topic not found"}), 404

# Start the Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=PORT)
  