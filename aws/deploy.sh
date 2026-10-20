#!/usr/bin/env bash
# Deploy StageSight to AWS with the AWS CLI (no Docker). Safe to re-run: updates what exists.
#   bash aws/deploy.sh                     # first deploy or code update
#   AGENT_BACKEND=bedrock bash aws/deploy.sh   # switch the agent to Claude on Bedrock
set -euo pipefail

AWS=${AWS:-aws}
REGION=${REGION:-us-east-2}   # account was created on the free plan, which pinned it to Ohio
export AWS_DEFAULT_REGION=$REGION AWS_REGION=$REGION   # every CLI call below targets this region
ACCOUNT=$($AWS sts get-caller-identity --query Account --output text)
BUCKET=${BUCKET:-stagesight-$ACCOUNT-$REGION}
ROLE=stagesight-lambda-role
FN_TICK=stagesight-scheduled
FN_API=stagesight-api
RULE=stagesight-every-15-min
AGENT_BACKEND=${AGENT_BACKEND:-rules}
AGENT_MODEL=${AGENT_MODEL:-global.anthropic.claude-opus-4-6-v1}
ZIP=build/stagesight_lambda.zip
# operator token for POST /approve: keep the deployed one, else make one (printed once below)
APPROVE_TOKEN=${APPROVE_TOKEN:-$($AWS lambda get-function-configuration --function-name "$FN_API"   --query Environment.Variables.APPROVE_TOKEN --output text 2>/dev/null | grep -v None || true)}
[ -z "$APPROVE_TOKEN" ] && APPROVE_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(18))") && NEW_TOKEN=1
# approved alerts and flags go out by email through SNS (the only path out of the system)
TOPIC_ARN=$($AWS sns create-topic --name stagesight-alerts --query TopicArn --output text)
if [ -n "${ALERT_EMAIL:-}" ] && ! $AWS sns list-subscriptions-by-topic --topic-arn "$TOPIC_ARN"      --query "Subscriptions[].Endpoint" --output text | grep -qw "$ALERT_EMAIL"; then
  $AWS sns subscribe --topic-arn "$TOPIC_ARN" --protocol email --notification-endpoint "$ALERT_EMAIL" >/dev/null
  echo "subscribed $ALERT_EMAIL to $TOPIC_ARN (confirm the email AWS sends)"
fi
ENV="Variables={ALERT_TOPIC_ARN=$TOPIC_ARN,STORE=s3:$BUCKET,UPSTREAM_SITE=01417500,AGENT_BACKEND=$AGENT_BACKEND,AGENT_MODEL=$AGENT_MODEL,BEDROCK_REGION=${BEDROCK_REGION:-us-east-1},APPROVE_TOKEN=$APPROVE_TOKEN,OPENCV_IO_MAX_IMAGE_PIXELS=40000000}"

echo "== account $ACCOUNT, region $REGION, bucket $BUCKET, agent $AGENT_BACKEND"

# 1. private bucket for results and code
if ! $AWS s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then $AWS s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else $AWS s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
         --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
  $AWS s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  echo "created bucket $BUCKET (private)"
fi

# 2. execution role: logs, this bucket only, Bedrock model calls, the alert topic
if ! $AWS iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  $AWS iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  $AWS iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "created role $ROLE"; sleep 10   # IAM propagation
fi
$AWS iam put-role-policy --role-name "$ROLE" --policy-name stagesight-access --policy-document "{
  \"Version\":\"2012-10-17\",\"Statement\":[
   {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/*\"},
   {\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\"],\"Resource\":\"arn:aws:s3:::$BUCKET\"},
   {\"Effect\":\"Allow\",\"Action\":[\"bedrock:InvokeModel\",\"bedrock:InvokeModelWithResponseStream\"],
    \"Resource\":[\"arn:aws:bedrock:*::foundation-model/anthropic.claude-*\",\"arn:aws:bedrock:*:$ACCOUNT:inference-profile/*anthropic.claude-*\"]},
   {\"Effect\":\"Allow\",\"Action\":[\"sns:Publish\"],\"Resource\":\"$TOPIC_ARN\"}]}"
ROLE_ARN=$($AWS iam get-role --role-name "$ROLE" --query Role.Arn --output text)

# 3. code
$AWS s3 cp "$ZIP" "s3://$BUCKET/code/stagesight_lambda.zip" --only-show-errors

deploy_fn () {   # name handler memory timeout
  if $AWS lambda get-function --function-name "$1" >/dev/null 2>&1; then
    $AWS lambda update-function-code --function-name "$1" --s3-bucket "$BUCKET" \
      --s3-key code/stagesight_lambda.zip >/dev/null
    $AWS lambda wait function-updated --function-name "$1"
    $AWS lambda update-function-configuration --function-name "$1" --handler "$2" \
      --memory-size "$3" --timeout "$4" --environment "$ENV" >/dev/null
  else
    $AWS lambda create-function --function-name "$1" --runtime python3.12 --architectures arm64 \
      --role "$ROLE_ARN" --handler "$2" --memory-size "$3" --timeout "$4" \
      --code S3Bucket="$BUCKET",S3Key=code/stagesight_lambda.zip --environment "$ENV" >/dev/null
  fi
  $AWS lambda wait function-updated --function-name "$1"
  echo "function $1 ready"
}
deploy_fn "$FN_TICK" handler.scheduled 1024 300   # peak use 295 MB; network-bound
deploy_fn "$FN_API" handler.api 1024 30
# a failed tick must not be retried: retries would repeat model calls and timeseries rows
$AWS lambda put-function-event-invoke-config --function-name "$FN_TICK" --maximum-retry-attempts 0 >/dev/null
# the public API runs OpenCV on /measure; cap its concurrency so traffic cannot run up cost
$AWS lambda put-function-concurrency --function-name "$FN_API" --reserved-concurrent-executions 10 >/dev/null

# 4. public Function URL for the dashboard and API; POST /approve needs the operator token
if ! $AWS lambda get-function-url-config --function-name "$FN_API" >/dev/null 2>&1; then
  $AWS lambda create-function-url-config --function-name "$FN_API" --auth-type NONE \
    --cors 'AllowOrigins=["*"],AllowMethods=["GET","POST"],AllowHeaders=["content-type","authorization"]' >/dev/null
  $AWS lambda add-permission --function-name "$FN_API" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
  $AWS lambda add-permission --function-name "$FN_API" --statement-id public-url-invoke \
    --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null 2>&1 || true
fi
URL=$($AWS lambda get-function-url-config --function-name "$FN_API" --query FunctionUrl --output text)

# 5. schedule: every 15 minutes
FN_ARN=$($AWS lambda get-function --function-name "$FN_TICK" --query Configuration.FunctionArn --output text)
RULE_ARN=$($AWS events put-rule --name "$RULE" --schedule-expression "rate(15 minutes)" --query RuleArn --output text)
$AWS events put-targets --rule "$RULE" --targets "Id=tick,Arn=$FN_ARN" >/dev/null
$AWS lambda add-permission --function-name "$FN_TICK" --statement-id eventbridge \
  --action lambda:InvokeFunction --principal events.amazonaws.com --source-arn "$RULE_ARN" >/dev/null 2>&1 || true

# 6. keep the API warm during judging: a GET every 5 minutes avoids the ~4 s cold start
API_ARN=$($AWS lambda get-function --function-name "$FN_API" --query Configuration.FunctionArn --output text)
WARM_ARN=$($AWS events put-rule --name stagesight-keep-warm --schedule-expression "rate(5 minutes)" --query RuleArn --output text)
printf '[{"Id":"warm","Arn":"%s","Input":"{\\"rawPath\\":\\"/latest\\",\\"requestContext\\":{\\"http\\":{\\"method\\":\\"GET\\"}}}"}]' "$API_ARN" > build/warm_targets.json
$AWS events put-targets --rule stagesight-keep-warm --targets file://build/warm_targets.json >/dev/null
$AWS lambda add-permission --function-name "$FN_API" --statement-id keep-warm   --action lambda:InvokeFunction --principal events.amazonaws.com --source-arn "$WARM_ARN" >/dev/null 2>&1 || true

echo "== deployed. Dashboard and API: $URL"
if [ -n "${NEW_TOKEN:-}" ]; then echo "== operator token for approvals (keep it private): $APPROVE_TOKEN"; fi
