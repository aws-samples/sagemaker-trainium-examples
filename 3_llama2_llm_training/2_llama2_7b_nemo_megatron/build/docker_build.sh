#!/usr/bin/env bash
# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.
# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
# set region

if [ "$#" -eq 4 ]; then
    dlc_account_id=$1
    region=$2
    image=$3
    tag=$4
else
    echo "usage: $0 <dlc-account-id> $1 <aws-region> $2 <image-repo> $3 <image-tag>"
    exit 1
fi

aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $dlc_account_id.dkr.ecr.$region.amazonaws.com
chmod +x build_and_push.sh; bash build_and_push.sh $dlc_account_id $region $image $tag 