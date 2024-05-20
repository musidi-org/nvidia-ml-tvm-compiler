from modal import Image, Stub, web_endpoint

gpu = 'T4'
stub = Stub(name=f'nvidia-ml-tvm-compiler-{gpu.lower()}')

tvmImage = Image.debian_slim(
  python_version='3.10'
).run_commands(
  [
    'pip install apache-tvm-cu116 -f https://tlcpack.ai/wheels'
  ]
)

@stub.function(image=tvmImage, container_idle_timeout=2, cpu=1, gpu=gpu)
@web_endpoint()
def my_function():
  import time
  now = time.time()
  print(time.time() - now)
  return {
    'duration': time.time() - now
  }
