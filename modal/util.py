import os
import time
import logging

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



def testModel(package, device):
  from tvm.driver import tvmc
  print(tvmc.run(
    package,
    device=device,
    repeat=10,
    number=10,
    benchmark=True,
    end_to_end=True
  ))
  
def incrementTuneModel(modelPath, target, tuningRecords):
  from tvm.driver import tvmc
  logging.getLogger('autotvm').setLevel(level=logging.ERROR)
  tvmc.tune(
    tvmc.load(modelPath),
    target=target,
    tuning_records = tuningRecords,
    prior_records = tuningRecords,
    trials=2000,
    timeout=1,
    parallel=16,
    enable_autoscheduler = True, # Auto-scheduler is faster than AutoTVM
  )
  logging.getLogger('autotvm').setLevel(level=logging.INFO)
  
def compileModel(model, target, tuningRecords, packagePath):
  from tvm.driver import tvmc
  return tvmc.compile(model, target=target, tuning_records=tuningRecords, package_path=packagePath, output_format='tar')

def tuneModel(modelPath, target, device):
  from tvm.driver import tvmc
  tuningRecords = f'package/{fileBaseName(modelPath)}.json'
  packagePath = f'package/{fileBaseName(modelPath)}.tar'
  createFile(tuningRecords)
  timeStamp = TimeStamp()
  
  model = tvmc.load(modelPath)
  timeStamp.stampPrint('LOAD MODEL')
  
  package =  compileModel(model, target, tuningRecords, packagePath)
  timeStamp.stampPrint('COMPILE OLD MODEL')
  
  testModel(package, device)
  oldPerformance = timeStamp.stampPrint('RUN OLD MODEL')
  
  incrementTuneModel(modelPath, target, tuningRecords)
  timeStamp.stampPrint('TUNING MODEL')

  package =  compileModel(model, target, tuningRecords, packagePath)
  timeStamp.stampPrint('COMPILE NEW MODEL')
  
  testModel(package, device)
  newPerformance = timeStamp.stampPrint('RUN NEW MODEL')

  modelSpeedup = oldPerformance / newPerformance
  print(f'IMPROVEMENT: x{modelSpeedup}')
  return modelSpeedup
