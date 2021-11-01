This directory contians an implementation of the Grey wolf optimization algorithm for hyperparameter tuning
in Optical-Rl-Gym.

- grey_wolf_deep_rmsa.py - Contains code for defining the objective function to optimize, as well as code for
environment creation and running the optimization. This is the entry point - the file you run.
    - make_environment() - Responsible for creating a Gym environment; you can configure logging here, as is 
    done using stable_baselines3.common.monitor.Monitor(). This function is run for each subprocess to create
    a new environment.
    - objective_function() - This is the function you want to optimize. The numeric (float) return value from
    this function will be used as the "fitness value" in the grey wolf optimization. The best wolves are those
    with the minimum fitness value (i.e. we are minimizing).
    - run() - Here is where you define the constants that will in turn define the range of search in each 
    dimension for the wolves. The best hyperparameters then will be inputs within the range that you have defined,
    which produce the lowest fitness value (output of objective_function). The selected topology is also defined
    in this method.

- grey_wolf.py - Contains a (class) definition for a grey wolf optimizer.

You will need to set the environment variable GWO_PROJECT_ROOT_DIR to the project root directory (/optical-rl-gym) 
before you run grey_wolf_deep_rmsa.py

Important inputs to the GreyWolfOptimizer class: -
    - dimensions: Number of dimensions that make up your input to the objective function 
        (how many hyperaparameters do you intend to optimize?)
    - intervals: Dictionary where the value is a list of 2 numbers defining the range to be searched.
        Each key-value pair corresponds to 1 dimension, so $dimensions should match len($intervals)
    - subprocesses: Number of subprocesses you want to use to run the hyperparameter optimization in 
        parallel. This should probably not exceed the number of CPU cores you have available.
    - num_wolves: Number of wolves to use in the search. This will increase the number of times
        the objective function is run (i.e. fitness value is evaluated).
        For example: 3 wolves * 100 max_iteration_number = 300 individual DeepRL training runs
    - max_iteration_number: Number of iterations allowed for the wolves to re-position
        themselves by searching for new hyperparameters.

Multiprocessing is implemented using SubProcEnv and VecExtractDictObs from stable_baselines3.common.vec_env.

Output monitor.csv files are written to a directory inside ./grey_wolf_deep_rmsa_ppo/{unix_timestamp}/

Note: the objective function takes wolf_number and iteration_number as inputs so that the logs for each 
training run can be written to the coressponding file with these labels.
(f"{logdir}/gwo_s{subprocess_index}_i{iteration_number}_w{wolf_number}_{uuid.uuid4()}")
