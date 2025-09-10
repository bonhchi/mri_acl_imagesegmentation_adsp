from abc import Abc, abstractmethod

class BaseAdapter:

    @abstractmethod
    def discover_records(self, root_dir):
        pass

    @abstractmethod
    def load_record(self, record):
        pass