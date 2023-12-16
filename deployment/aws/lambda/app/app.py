import pickle
import json

with open('fcst.pkl', 'rb') as f:
    fcst = pickle.load(f)

def lambda_handler(event, context):
    preds = fcst.predict(h=event["h"], ids=event.get("ids", None))
    preds['ds'] = preds['ds'].astype(str)
    preds = preds.to_dict(orient='records')
    return {
        "statusCode": 200,
        "body": {"predictions": preds},
    }
