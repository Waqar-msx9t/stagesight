"""Run the COOL benchmark on AWS Graviton4, end to end, without SSH.

1. packs the reader, the Hancock site and the 184 frames of 21 Jul - 1 Aug 2026 and uploads them to S3;
2. starts one EC2 instance from the COOL AMI (Cloud Optimized OpenCV for AWS Graviton4) whose
   user-data runs scripts/cool_bench.py twice on the same machine: with COOL, then with pip
   opencv-python-headless 5.0.0.93 (the build StageSight uses on Lambda);
3. the instance uploads the results and the log, then shuts down (shutdown = terminate);
4. this script downloads the results to results/cool/.

Needs: an active AWS session (aws login), a subscription to the COOL Marketplace listing, and the
frames in data/hancock (scripts/download_site.py).
Usage:  python scripts/cool_run.py [--instance-type c8g.xlarge] [--ami ami-005a9f90c70ba8d9b]
"""
import argparse
import csv
import io
import os
import tarfile
import time

import boto3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGION = "us-east-2"
COOL_AMI = "ami-005a9f90c70ba8d9b"     # Cloud Optimized OpenCV 3.1 (OpenCV 5) for Graviton4, us-east-2
USER_DATA = """#!/bin/bash
exec > /var/log/stagesight-bench.log 2>&1
up() { tar czf /root/res.tgz -C /root out /var/log/stagesight-bench.log; curl -s -X PUT --upload-file /root/res.tgz "{put}"; }
trap 'up; shutdown -h now' EXIT
( sleep 2700; shutdown -h now ) &        # hard limit: the instance is gone after 45 min whatever happens
mkdir -p /root/w /root/out && cd /root/w
curl -s -o pkg.tgz "{get}" && tar xzf pkg.tgz   # presigned URLs stay out of the log (no set -x before here)
set -x
lscpu > /root/out/lscpu.txt; nproc >> /root/out/lscpu.txt
source /opt/cool/venvs/python_3.12/bin/activate
python -c "import cv2, numpy; print(cv2.__version__, cv2.__file__, numpy.__version__)" > /root/out/cool_import.txt
python -X faulthandler scripts/cool_bench.py --frames frames --out /root/out/cool.json --repeats 2
deactivate
apt-get update -q && apt-get install -y -q python3.12-venv
unset PYTHONPATH LD_LIBRARY_PATH          # COOL's activate exports these and deactivate keeps them
python3.12 -m venv /root/pipcv
/root/pipcv/bin/pip install -q opencv-python-headless==5.0.0.93 numpy==2.5.4
/root/pipcv/bin/python -X faulthandler scripts/cool_bench.py --frames frames --out /root/out/pip.json --repeats 2
source /opt/cool/venvs/python_3.12/bin/activate
python -X faulthandler scripts/cool_bench.py --frames frames --out /root/out/cool_again.json --repeats 2
"""
# --scale: archive backfill with 1-16 worker processes for both builds, and the DNN crash, captured
SCALE = """
if [ -d frames_all ]; then
  python scripts/cool_scale.py --frames frames_all --out /root/out/scale_cool.json --workers 1 4 8 16
  python -X faulthandler scripts/cool_dnn_repro.py > /root/out/dnn_cool.txt 2>&1; echo "exit $?" >> /root/out/dnn_cool.txt
  deactivate; unset PYTHONPATH LD_LIBRARY_PATH
  /root/pipcv/bin/python scripts/cool_scale.py --frames frames_all --out /root/out/scale_pip.json --workers 1 4 8 16
  /root/pipcv/bin/python -X faulthandler scripts/cool_dnn_repro.py > /root/out/dnn_pip.txt 2>&1; echo "exit $?" >> /root/out/dnn_pip.txt
  ldd --version | head -1 > /root/out/glibc.txt
fi
"""


def package(scale: bool = False) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ("stagesight", "sites", "models"):
            tar.add(os.path.join(ROOT, d), d, filter=lambda t: None if "__pycache__" in t.name else t)
        for f in ("cool_bench.py", "cool_scale.py", "cool_dnn_repro.py"):
            tar.add(os.path.join(ROOT, "scripts", f), "scripts/" + f)
        rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "hancock", "images.csv"))))
        for r in rows:
            p = os.path.join(ROOT, os.path.normpath(r["path"]))
            if "2026-07-21" <= r["time_utc"][:10] <= "2026-08-01":
                tar.add(p, "frames/" + os.path.basename(p))
            if scale:                                   # the whole 90-day archive
                tar.add(p, "frames_all/" + os.path.basename(p))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="c8g.xlarge")
    ap.add_argument("--ami", default=COOL_AMI)
    ap.add_argument("--scale", action="store_true", help="also batch the whole archive with 1-16 workers")
    a = ap.parse_args()
    account = boto3.client("sts").get_caller_identity()["Account"]
    bucket = f"stagesight-{account}-{REGION}"                      # the bucket aws/deploy.sh creates
    s3, ec2 = boto3.client("s3", region_name=REGION), boto3.client("ec2", region_name=REGION)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    pkg_key, res_key = f"cool/{stamp}/pkg.tgz", f"cool/{stamp}/results.tgz"
    s3.put_object(Bucket=bucket, Key=pkg_key, Body=package(a.scale))
    get = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": pkg_key}, ExpiresIn=7200)
    put = s3.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": res_key}, ExpiresIn=7200)
    r = ec2.run_instances(ImageId=a.ami, InstanceType=a.instance_type, MinCount=1, MaxCount=1,
                          UserData=(USER_DATA + (SCALE if a.scale else "")).replace("{get}", get).replace("{put}", put),
                          InstanceInitiatedShutdownBehavior="terminate",
                          TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "stagesight-cool-bench"}]}])
    iid = r["Instances"][0]["InstanceId"]
    print("launched", iid, a.instance_type, a.ami, flush=True)
    out_dir = os.path.join(ROOT, "results", "cool", a.instance_type)      # one folder per instance type
    os.makedirs(out_dir, exist_ok=True)
    try:
        deadline = time.time() + 3600
        while time.time() < deadline:
            time.sleep(30)
            try:
                body = s3.get_object(Bucket=bucket, Key=res_key)["Body"].read()
            except s3.exceptions.NoSuchKey:
                state = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
                print("waiting,", state, flush=True)
                continue
            with tarfile.open(fileobj=io.BytesIO(body)) as tar:
                tar.extractall(out_dir, filter="data")
            print("results in", out_dir, flush=True)
            break
    finally:                                    # never leave an instance or the uploads behind
        ec2.terminate_instances(InstanceIds=[iid])
        for key in (pkg_key, res_key):
            s3.delete_object(Bucket=bucket, Key=key)
        print("terminated", iid, "and removed the S3 objects", flush=True)


if __name__ == "__main__":
    main()
