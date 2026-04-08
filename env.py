import pandas as pd
import io
from models import Action, Observation

class DataJanitorEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Create a small "dirty" dataframe
        data = {
            'date': ['2023-01-01', '02/01/2023', None, '2023-01-04'],
            'sales': [100, None, 150, 200],
            'id': [1, 2, 2, 3]
        }
        self.df = pd.DataFrame(data)
        self.steps = 0
        return self._get_obs("Environment Reset. Fix the data.")

    def _get_obs(self, msg):
        return Observation(
            data_preview=self.df.to_string(),
            missing_counts=self.df.isnull().sum().to_string(),
            status_message=msg
        )

    def step(self, action: Action):
        self.steps += 1
        reward = 0.0
        done = False
        
        if action.command == "impute_mean" and action.column == "sales":
            self.df['sales'] = self.df['sales'].fillna(self.df['sales'].mean())
            reward = 0.3
        elif action.command == "drop_duplicates":
            self.df = self.df.drop_duplicates()
            reward = 0.3
        elif action.command == "submit":
            done = True
            # Final Grader Logic: Are there any nulls or duplicates left?
            if not self.df.isnull().values.any() and not self.df.duplicated().any():
                reward = 1.0
            else:
                reward = 0.1

        return self._get_obs(f"Action {action.command} applied"), reward, done, {}
