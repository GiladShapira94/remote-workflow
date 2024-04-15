

import mlrun

def pipeline():
    project = mlrun.get_current_project()
    # Fetch the data
    dataset = project.get_artifact("data")
    job_function_run = project.run_function('fetch-data',
        inputs={'dataset': dataset.target_path},
        outputs=["dataset"])

    # Train the model
    trainer_run = project.run_function("trainer",inputs = {"dataset":job_function_run.outputs['dataset']},
                                       params = {"n_estimators": 100, "learning_rate": 1e-1, "max_depth": 3},
                                      outputs=["model"])


    # Deploy the model
    project.run_function("predict",params={"model":trainer_run.outputs["model"]},returns=["predictions"])
