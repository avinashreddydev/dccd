from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import json
from typing import Iterator

class BaseLoader(ABC):
    def __init__(self):
        self.input_column : str = None
        self.output_column : str = None
        self.data : List[str] = None
        self.ground_truths : List[Dict] = None
        self.task_name : str = None


    @abstractmethod
    def load_data(self) -> List[Dict]:
        pass


    @abstractmethod
    def get_json_schema(self):
        pass
    

    def __getitem__(self, index: int) -> Dict:
        return {
            "input": self.data[index],
            "ground_truth": self.ground_truths[index]
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self) -> Iterator[Dict]:
        return iter(self.data)
    
    def __next__(self) -> Dict:
        return next(self.data)
    

    def is_json(self, string: str) -> bool:
        try:
            json.loads(string)
            return True
        except:
            return False
    
