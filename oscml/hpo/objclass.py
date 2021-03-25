import torch

class ObjectiveTask:
    def __init__(self, funcHandle=None, objParamsKey=None, extArgs=[]):
        self.funcHandle = funcHandle
        self.extArgs = extArgs
        self.objParamsKey = objParamsKey

    def run(self, *intArgs):
        return self.funcHandle(*intArgs, *self.extArgs)

class Objective:
    def __init__(self, modelName=None,
                 data=None, config=None, logFile= None, logDir=None,
                 logHead=None):
        self.modelName = modelName
        self.data = data
        self.objConfig = {
            'config': config,
            "log_file": logFile,
            "log_dir": logDir,
            "log_head": logHead
        }
        self.objParams = {}

        #self.processed_data = None
        self.trial = None
        self.obj_val = None
        self.model = None
        self.modelCreator = None
        self.modelTrainer = None
        self.preModelCreateTasks = []
        self.postModelCreateTasks = []
        self.postTrainingTasks = []

    def addPreModelCreateTask(self, funcHandle, objParamsKey, extArgs=[]):
        self.preModelCreateTasks.append(ObjectiveTask(funcHandle=funcHandle, objParamsKey=objParamsKey, extArgs=extArgs))

    def addPostModelCreateTask(self, funcHandle, objParamsKey, extArgs=[]):
        self.postModelCreateTasks.append(ObjectiveTask(funcHandle=funcHandle, objParamsKey=objParamsKey, extArgs=extArgs))

    def addPostTrainingTask(self, funcHandle, objParamsKey, extArgs=[]):
        self.postTrainingTasks.append(ObjectiveTask(funcHandle=funcHandle, objParamsKey=objParamsKey, extArgs=extArgs))

    def setModelCreator(self, funcHandle, extArgs=[]):
        self.modelCreator = ObjectiveTask(funcHandle=funcHandle, extArgs=extArgs)

    def setModelTrainer(self, funcHandle, extArgs=[]):
        self.modelTrainer = ObjectiveTask(funcHandle=funcHandle, extArgs=extArgs)

    def setDataPreprocFunc(self, funcHandle):
        self.dataPreprocFunc = funcHandle

    def _doTraining(self):
        self.obj_val = self.modelTrainer.run(self.trial, self.model, self.data, self.objConfig, self.objParams)

    def _createModel(self):
        self.model = self.modelCreator.run(self.trial, self.data, self.objConfig, self.objParams)

    def __call__(self, trial):
        self._releaseMemory()
        self._setTrial(trial)
        self._doPreModelCreateTasks()
        self._createModel()
        self._doPostModelCreateTasks()
        self._doTraining()
        self._doPostTrainingTasks()
        return self.obj_val

    def _setTrial(self, trial):
        self.trial = trial

    def _doPreModelCreateTasks(self):
        for task in self.preModelCreateTasks:
            task_result= task.run(self.trial, self.data, self.objConfig, self.objParams)
            if task.objParamsKey is not None:
                self.objParams[task.objParamsKey] = task_result

    def _doPostModelCreateTasks(self):
        for task in self.postModelCreateTasks:
            task_result = task.run(self.trial, self.model, self.data, self.objConfig, self.objParams)
            if task.objParamsKey is not None:
                self.objParams[task.objParamsKey] = task_result

    def _doPostTrainingTasks(self):
        for task in self.postTrainingTasks:
            task_result = task.run(self.trial, self.model, self.data, self.objConfig, self.objParams)
            if task.objParamsKey is not None:
                self.objParams[task.objParamsKey] = task_result

    @staticmethod
    def _releaseMemory():
        torch.cuda.empty_cache()