from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from datetime import datetime
import pandas as pd
from mlrun.frameworks.sklearn import SKLearnModelHandler 
def generate_random_rows(num_rows, mean=0, std=1):
    # Provided row as a reference for mean and standard deviation
    ref_row = [13.71, 20.83, 90.2, 577.9, 0.1189, 0.1645, 0.09366, 0.05985, 0.2196, 0.07451, 
               0.5835, 1.377, 3.856, 50.96, 0.008805, 0.03029, 0.02488, 0.01448, 0.01486, 
               0.005412, 17.06, 28.14, 110.6, 897.0, 0.1654, 0.3682, 0.2678, 0.1556, 
               0.3196, 0.1151]
    
    # Generate random numbers based on a normal distribution with the same mean and std as the reference row
    random_rows = np.random.normal(loc=mean, scale=std, size=(num_rows, len(ref_row)))
    
    # Add the reference row to each random row to make them similar
    random_rows += np.array(ref_row)
    
    return random_rows

def predict(context,model: str):
    model_handler = SKLearnModelHandler(model_path=model)
    model_handler.load()
    context.logger.info("Load the Model")
    random_to_predict = generate_random_rows(10)
    context.logger.info("Generated inputs for predicitions")
    predictions = model_handler.model.predict(random_to_predict)
    d = {"timestemp":str(datetime.now())*len(random_to_predict),"predictions":predictions}
    context.logger.info(f"Predictions={predictions}")
    df = pd.DataFrame(d)
    return df
