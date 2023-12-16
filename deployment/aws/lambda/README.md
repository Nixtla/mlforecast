# AWS Lambda deployment
This contains a template to deploy mlforecast in AWS Lambda using a docker image. 

## Steps
1. Update the `app/requirements.txt` file. If you want to use the dependencies you currently have installed in your environment you can use `pip freeze > app/requirements.txt`
2. Change the `train.py` file with your training script and run it. Note that it produces an artifact that is saved to `app/fcst.pkl`, this is the serialized forecast object and it will be copied into the docker image.
3. Update the `app/app.py` file with your application logic.
6. Update the `event.json` file with a sample event that the lambda function will recieve.
4. Verify if you have AWS SAM installed with `sam --version`. If you don't have it you can install it with `pip install aws-sam-cli`
5. Run `sam build`. This will create the docker image.
7. Test the application locally by running `sam local invoke PredictFunction -e event.json`
8. Run `sam deploy`
