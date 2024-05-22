from tvm.driver import tvmc

def autoTune(modelPath, target):
  tuning_records = f'logs/{modelPath}.json'
  model = tvmc.load(modelPath)
  tuning_records = tvmc.tune(
    model,
    target=target,
    enable_autoscheduler = True, # Auto-scheduler is faster than AutoTVM
    trials=500,
    timeout=1,
    prior_records = tuning_records,
    tuning_records = tuning_records
  )
  
if __name__ == "__main__":
  autoTune('my_model.onnx', 'llvm')
