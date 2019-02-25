class BaseModel:
    def __init__(self, architecture, task_info):
        pass

    def train(self, train_data, valid_data, configs, save_history, path, verbose):
        raise NotImplementedError

    def eval(self, data):
        raise NotImplementedError

    def save(self, path):
        pass

    def load(self, path):
        pass
