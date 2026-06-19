$ECR = "351998451865.dkr.ecr.ap-northeast-2.amazonaws.com/roadaccident:latest"
$CLUSTER = "roadaccident-cluster"
$SERVICE = "roadaccident-svc"
$REGION = "ap-northeast-2"

Write-Host "==> ECR login" -ForegroundColor Cyan
cmd /c "aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin 351998451865.dkr.ecr.ap-northeast-2.amazonaws.com"
if (-not $?) { exit 1 }

Write-Host "==> Docker build" -ForegroundColor Cyan
docker build -t roadaccident .
if (-not $?) { exit 1 }

Write-Host "==> Push to ECR" -ForegroundColor Cyan
docker tag roadaccident:latest $ECR
docker push $ECR
if (-not $?) { exit 1 }

Write-Host "==> Trigger ECS rolling deploy" -ForegroundColor Cyan
aws ecs update-service --cluster $CLUSTER --service $SERVICE --force-new-deployment --region $REGION --output text --query "service.deployments[0].status"

Write-Host "Done. Check status:" -ForegroundColor Green
Write-Host "  aws ecs describe-services --cluster $CLUSTER --services $SERVICE --region $REGION --query `"services[0].{Running:runningCount,Pending:pendingCount}`" --output table"
