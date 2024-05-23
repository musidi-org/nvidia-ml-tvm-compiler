from modal import Image, App, web_endpoint
from dotenv import load_dotenv
load_dotenv()
import os

from util import tuneModel
import package

app = App(name=os.environ.get('APP_NAME'))
# Python3.10 highest support version for packages
baseImage = Image.debian_slim(python_version='3.10').pip_install('onnx==1.16.0', 'xgboost==1.5.2', 'python-dotenv==1.0.1')
tvmImageCpu = baseImage.run_commands('pip install apache-tvm==0.14.dev273')
tvmImageGpu = baseImage.run_commands(['pip install apache-tvm-cu116 -f https://tlcpack.ai/wheels'])

timeout = 60 * 60 * 12

# Specify functions based on container specs
@app.function(image=tvmImageCpu, container_idle_timeout=2, timeout=timeout, cpu=1)
@web_endpoint()
def cpu_1():
  return tuneModel('package/test.onnx', 'llvm', 'cpu')

@app.function(image=tvmImageGpu, container_idle_timeout=2, timeout=timeout, cpu=1, gpu='t4')
@web_endpoint()
def t4_1():
  return tuneModel('package/test.onnx', 'llvm', 'cpu')
