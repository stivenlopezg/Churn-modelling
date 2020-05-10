import boto3


def lambda_handler(event, context):
    client = boto3.client('sagemaker-runtime', 'us-east-1')
    score_endpoint = 'churn-score-pdn'
    response = client.invoke_endpoint(EndpointName=score_endpoint,
                                      Body=event['Body'],
                                      ContentType='text/csv',
                                      Accept='text/csv')
    score = 1 if float(response['Body'].read().decode('utf-8')) >= 0.5 else 0
    return {
        'statusCode': 200,
        'headers': {'ContentType': 'text/csv',
                    'Acces-Control-Allow-Origin': '*'},
        'body': score
    }
