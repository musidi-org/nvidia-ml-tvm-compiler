import os
import time

def createFile(filePath):
  if not os.path.exists(filePath): 
    with open(filePath, 'w') as file: 
      file.write('')



def fileBaseName(filePath):
  fileName = os.path.basename(filePath)
  return os.path.splitext(fileName)[0]



class TimeStamp:
  def __init__(self) -> None:
    self.startTime = time.time()
    
  def stamp(self):
    timeGap = time.time() - self.startTime
    self.startTime = time.time()
    return timeGap
  
  def stampPrint(self, log):
    timeGap = self.stamp()
    print(f'\n\n\n{log}: ', timeGap, '\n\n\n')
    return timeGap



def tuneModel(modelPath, target, device):
  from tvm.driver import tvmc
  timeStamp = TimeStamp()
  
  model = tvmc.load(modelPath)
  timeStamp.stampPrint('LOAD MODEL')
  
  package = tvmc.compile(model, target=target)
  timeStamp.stampPrint('COMPILE OLD MODEL')
  
  print(tvmc.run(
    package,
    device=device,
    repeat=10,
    number=1
  ))
  oldPerformance = timeStamp.stampPrint('RUN OLD MODEL')
  
  tuningRecords = f'package/{fileBaseName(modelPath)}.json'
  createFile(tuningRecords)
  tvmc.tune(
    tvmc.load(modelPath),
    target=target,
    enable_autoscheduler = True, # Auto-scheduler is faster than AutoTVM
    trials=500,
    prior_records = tuningRecords,
    tuning_records = tuningRecords
  )
  timeStamp.stampPrint('TUNING MODEL')
  
  package = tvmc.compile(model, target=target, tuning_records=tuningRecords)
  timeStamp.stampPrint('COMPILE NEW MODEL')
  
  print(tvmc.run(
    package,
    device=device,
    repeat=10,
    number=10
  ))
  newPerformance = timeStamp.stampPrint('RUN NEW MODEL')
  
  modelSpeedup = newPerformance / oldPerformance
  print(f'IMPROVEMENT: x{modelSpeedup}')
  return modelSpeedup
