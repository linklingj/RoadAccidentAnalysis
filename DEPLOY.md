# 배포 가이드

## 환경 변수
```
AWS_ACCOUNT = 351998451865
AWS_REGION  = ap-northeast-2
ECR_URI     = 351998451865.dkr.ecr.ap-northeast-2.amazonaws.com/roadaccident:latest
```

## 코드 수정 후 재배포 (4단계)

```powershell
# 1. ECR 로그인
cmd /c "aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 351998451865.dkr.ecr.ap-northeast-2.amazonaws.com"

# 2. 이미지 빌드
docker build -t roadaccident .

# 3. ECR 푸시
docker tag roadaccident:latest 351998451865.dkr.ecr.ap-northeast-2.amazonaws.com/roadaccident:latest
docker push 351998451865.dkr.ecr.ap-northeast-2.amazonaws.com/roadaccident:latest

# 4. ECS 새 배포 트리거
aws ecs update-service `
    --cluster roadaccident-cluster `
    --service roadaccident-svc `
    --force-new-deployment `
    --region ap-northeast-2
```

ECS가 자동으로 구 컨테이너를 교체합니다 (롤링 배포, 약 3~5분 소요).

## 배포 상태 확인

```powershell
# Running 카운트가 1이 되면 완료
aws ecs describe-services `
    --cluster roadaccident-cluster `
    --services roadaccident-svc `
    --region ap-northeast-2 `
    --query "services[0].{Running:runningCount,Pending:pendingCount}" `
    --output table
```

## 접속 URL
```
http://roadaccident-alb-1568059849.ap-northeast-2.elb.amazonaws.com
```

## 비용 절약 (사용 안 할 때)

```powershell
# Task 중단 (ALB·클러스터는 유지, Fargate 비용 없음)
aws ecs update-service --cluster roadaccident-cluster --service roadaccident-svc --desired-count 0 --region ap-northeast-2

# Task 재시작
aws ecs update-service --cluster roadaccident-cluster --service roadaccident-svc --desired-count 1 --region ap-northeast-2
```

## 로그 확인

```powershell
aws logs tail /ecs/roadaccident --follow --region ap-northeast-2
```
