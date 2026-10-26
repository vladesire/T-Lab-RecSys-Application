"""Protocol example only. Replace compute_result; no infrastructure imports."""
import argparse
import json
from pathlib import Path

def compute_result(request, output):
    raise NotImplementedError("Implement branch computation and result.json schema")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--request",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    request=json.loads(Path(a.request).read_text())
    output=Path(a.output)
    output.mkdir(parents=True,exist_ok=True)
    result=compute_result(request,output)
    (output/"result.json").write_text(json.dumps(result,allow_nan=False,sort_keys=True,indent=2))

if __name__=="__main__": main()
