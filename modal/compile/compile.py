import os
from tvm.driver import tvmc

def fileBaseName(filePath):
  fileName = os.path.basename(filePath)
  return os.path.splitext(fileName)[0]

def autoTune(modelPath, **kwargs):
  tuningRecords = f'compile/{fileBaseName(modelPath)}.json'
  tvmc.tune(
    tvmc.load(modelPath),
    target=kwargs.pop('target', 'llvm'),
    enable_autoscheduler = True, # Auto-scheduler is faster than AutoTVM
    trials=kwargs.pop('trails', 500),
    prior_records = tuningRecords,
    tuning_records = tuningRecords
  )
  
if __name__ == "__main__":
  autoTune('my_model.onnx')
