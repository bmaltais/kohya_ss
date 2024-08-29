#!/bin/bash

fastapi run app/main.py &

# Log a message to stdout
echo "Running SQS handler..."

# Start the SQS handler script
python3 sqs_handler.py &

# Keep the script running
tail -f /dev/null
